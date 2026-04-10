# GPT-2 with Rotary Position Embedding (RoPE)
# Architecture strictly follows Appendix F of:
#   "Physics of Language Models: Part 2.1" (Ye et al., 2024)
#
# Naming convention: GPT2-ℓ-h  →  ℓ layers, h heads, 64h hidden dim
#
# Size-1 models (~GPT2-small, 117M params):
#   GPT2-4-21, GPT2-8-15, GPT2-12-12, GPT2-16-10, GPT2-20-9
# Size-2 models (~2x GPT2-small):
#   GPT2-4-30, GPT2-8-21, GPT2-12-17, GPT2-16-15, GPT2-20-13

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(head_dim: int, max_seq_len: int, base: float = 10000.0):
    """Precompute RoPE cos/sin tables using real arithmetic (torch.compile friendly)."""
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, theta)               # (T, head_dim/2)
    # Store as (T, head_dim/2, 2): [..., 0]=cos, [..., 1]=sin
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # real, no complex ops


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor):
    """
    Apply RoPE to query and key tensors using real arithmetic.
    xq, xk:    (B, T, n_heads, head_dim)
    freqs_cis: (T, head_dim/2, 2)  — [...,0]=cos, [...,1]=sin
    """
    cos = freqs_cis[..., 0].unsqueeze(0).unsqueeze(2)  # (1, T, 1, head_dim/2)
    sin = freqs_cis[..., 1].unsqueeze(0).unsqueeze(2)

    # Split interleaved pairs: positions 0,2,4,... and 1,3,5,...
    xq_e, xq_o = xq[..., 0::2].float(), xq[..., 1::2].float()
    xk_e, xk_o = xk[..., 0::2].float(), xk[..., 1::2].float()

    # Complex rotation: (a + ib)(cos θ + i sin θ) = (a cosθ − b sinθ) + i(a sinθ + b cosθ)
    xq_out = torch.stack([xq_e * cos - xq_o * sin,
                          xq_e * sin + xq_o * cos], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_e * cos - xk_o * sin,
                          xk_e * sin + xk_o * cos], dim=-1).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    # Architecture (GPT2-ℓ-h: n_layer=ℓ, n_head=h, n_embd=64*h)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768        # must equal 64 * n_head
    # Vocabulary / sequence
    vocab_size: int = 50257  # GPT-2 tokenizer (token ids 0..50256)
    block_size: int = 768    # context length for iGSM-med pretraining
    # Regularization
    dropout: float = 0.0
    bias: bool = False       # GPT-2 style: no bias in Linear/LayerNorm
    # RoPE
    rope_base: float = 10000.0
    rope_max_seq_len: int = 4096  # precompute up to this length

    def __post_init__(self):
        assert self.n_embd == 64 * self.n_head, (
            f"n_embd ({self.n_embd}) must equal 64 * n_head ({self.n_head})"
        )


# Predefined configs matching the paper's model variants
# Usage: cfg = MODEL_CONFIGS["GPT2-12-12"]
MODEL_CONFIGS = {
    # Size-1 (≈ GPT2-small, 117M)
    "GPT2-4-21":  GPTConfig(n_layer=4,  n_head=21, n_embd=1344),
    "GPT2-8-15":  GPTConfig(n_layer=8,  n_head=15, n_embd=960),
    "GPT2-12-12": GPTConfig(n_layer=12, n_head=12, n_embd=768),   # default
    "GPT2-16-10": GPTConfig(n_layer=16, n_head=10, n_embd=640),
    "GPT2-20-9":  GPTConfig(n_layer=20, n_head=9,  n_embd=576),
    # Size-2 (≈ 2x GPT2-small)
    "GPT2-4-30":  GPTConfig(n_layer=4,  n_head=30, n_embd=1920),
    "GPT2-8-21":  GPTConfig(n_layer=8,  n_head=21, n_embd=1344),
    "GPT2-12-17": GPTConfig(n_layer=12, n_head=17, n_embd=1088),
    "GPT2-16-15": GPTConfig(n_layer=16, n_head=15, n_embd=960),
    "GPT2-20-13": GPTConfig(n_layer=20, n_head=13, n_embd=832),
}


# ---------------------------------------------------------------------------
# Model layers
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                past_kv=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # reshape to (B, T, n_head, head_dim) for RoPE, then to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        q = q.transpose(1, 2)   # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Append cached keys/values from previous steps
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)

        # When using KV cache for single-token generation, all keys are past tokens —
        # no future tokens to mask, so is_causal=False is correct.
        is_causal = past_kv is None
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor,
                past_kv=None):
        attn_out, new_kv = self.attn(self.ln_1(x), freqs_cis, past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


# ---------------------------------------------------------------------------
# GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (input embedding ↔ output projection)
        self.transformer.wte.weight = self.lm_head.weight

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.rope_max_seq_len,
            config.rope_base,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections (GPT-2 paper §2)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                past_kvs=None, start_pos: int = 0,
                use_cache: bool = False):
        """
        idx:       (B, T)  token ids
        targets:   (B, T)  shifted token ids for LM loss (optional)
        past_kvs:  list of (k, v) per layer from a previous forward call (KV cache)
        start_pos: position offset for RoPE when using KV cache
        use_cache: if True, return (logits, loss, new_kvs); else return (logits, loss)

        Training: model(x, y)                    → (logits, loss)
        Inference with KV cache:
          prefill: model(prompt, use_cache=True)  → (logits, None, new_kvs)
          decode:  model(tok, past_kvs=kvs,
                         start_pos=n, use_cache=True) → (logits, None, new_kvs)
        """
        B, T = idx.shape
        assert start_pos + T <= self.config.block_size, (
            f"start_pos({start_pos}) + T({T}) exceeds block_size({self.config.block_size})"
        )

        x = self.transformer.wte(idx)     # (B, T, n_embd)
        x = self.transformer.drop(x)

        freqs_cis = self.freqs_cis[start_pos: start_pos + T]   # (T, head_dim/2)

        new_kvs = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, freqs_cis, past_kv=past_kv)
            new_kvs.append(new_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            x_2d   = x.view(-1, x.size(-1))          # (B*T, n_embd)
            tgt_1d = targets.view(-1)                 # (B*T,)
            logits_2d = self.lm_head(x_2d)            # (B*T, vocab)
            loss = F.cross_entropy(logits_2d, tgt_1d, ignore_index=-1)
            logits = logits_2d.view(x.size(0), x.size(1), -1)[:, -1:, :]
        else:
            logits = self.lm_head(x)                  # (B, T, vocab_size)
            loss   = None

        if use_cache:
            return logits, loss, new_kvs
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: Optional[int] = None):
        """Autoregressive generation with KV cache (greedy or top-k sampling)."""
        # Prefill: process the full prompt once, collect KV cache
        prompt_len = idx.size(1)
        logits, _, past_kvs = self(idx, use_cache=True, start_pos=0)
        cur_len = prompt_len

        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits_last, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if cur_len >= self.config.block_size:
                break
            # Decode one token using KV cache (O(1) attention per step)
            logits, _, past_kvs = self(
                idx_next, past_kvs=past_kvs, start_pos=cur_len, use_cache=True
            )
            cur_len += 1
        return idx

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wte.weight.numel()
        return n
