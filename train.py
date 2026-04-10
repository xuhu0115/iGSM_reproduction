# Pretraining script for iGSM reproduction (Phase 2)
# Strictly follows Appendix F.1 of:
#   "Physics of Language Models: Part 2.1" (Ye et al., 2024)
#
# Key settings from the paper:
#   Optimizer  : AdamW, β=(0.9, 0.98), fp16, cosine LR decay to 0.01x, warmup=1000
#   iGSM-med   : lr=2e-3, wd=0.05, batch=512, ctx=768,  steps=100k
#   iGSM-hard  : lr=2e-3, wd=0.03, batch=256, ctx=1024, steps=200k
#   Main model : GPT2-12-12 (12-layer, 12-head, 768-dim)
#
# Gradient accumulation is supported via --micro_batch.
# global_batch = micro_batch × accum_steps × world_size
# accum_steps is inferred automatically to match the paper's global batch size.
#
# Usage (single GPU, auto gradient accumulation to match batch=512):
#   python train.py --dataset med --micro_batch 32
#
# Usage (2 GPU with torchrun, each GPU micro_batch=64 → global=512 via 4 accum):
#   torchrun --nproc_per_node=2 train.py --dataset med --micro_batch 64

import os
import sys
import math
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import GPT, GPTConfig, MODEL_CONFIGS

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TokenBinDataset(Dataset):
    """
    Streams fixed-length chunks from a flat uint16 binary file.
    File layout: concatenated token IDs produced by generate_dataset.py.
    """
    def __init__(self, bin_path: str, block_size: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.n = (len(self.data) - 1) // block_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.data[start: start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class OnTheFlyDataset(Dataset):
    """
    Generates iGSM problems on-the-fly (matches the paper's training setup).
    A single 'epoch' is a fixed number of steps; call reset() to re-shuffle.
    """
    def __init__(self, dataset_type: str, block_size: int,
                 steps_per_epoch: int, batch_size: int, p_format: str = "pq"):
        from data_gen.pretrain.id_gen import IdGen
        from const.params import train_bin

        self.block_size = block_size
        self.total_samples = steps_per_epoch * batch_size
        self.p_format = p_format

        max_op   = 15 if dataset_type == "med" else 21
        max_edge = 20 if dataset_type == "med" else 28
        self.id_gen   = IdGen(max_op=max_op, max_edge=max_edge,
                              perm_level=5, detail_level=0)
        self.ava_hash = train_bin

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        import random
        # Generate problems until we fill one context window
        tokens = []
        while len(tokens) < self.block_size + 1:
            try:
                # Sample op uniformly and set both op and op_ so that
                # gen_param_light uses the op-aware branch (s = self.op),
                # ensuring high-op samples can actually be generated.
                new_op = random.randint(1, self.id_gen.max_op)
                self.id_gen.op  = new_op
                self.id_gen.op_ = new_op
                self.id_gen.perm_level_ = (random.randint(0, 6)
                                           if self.id_gen.perm_level is None
                                           else self.id_gen.perm_level)
                self.id_gen.detail_level_ = (random.randint(0, 11)
                                             if self.id_gen.detail_level is None
                                             else self.id_gen.detail_level)
                self.id_gen.gen_prob(self.ava_hash, p_format=self.p_format)
                tokens.extend(self.id_gen.token_id)
            except Exception:
                continue
        tokens = tokens[:self.block_size + 1]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:],  dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int,
           peak_lr: float, min_lr_ratio: float = 0.01) -> float:
    """Cosine decay with linear warmup, decays to peak_lr * min_lr_ratio."""
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    if step >= total_steps:
        return peak_lr * min_lr_ratio
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * coeff)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # ---- Hyperparameters (paper Table in Appendix F.1) -------------------
    if args.dataset == "med":
        peak_lr    = 2e-3
        weight_decay = 0.05
        batch_size = 512
        block_size = 768
        total_steps = 100_000
    else:  # hard
        peak_lr    = 2e-3
        weight_decay = 0.03
        batch_size = 256
        block_size = 1024
        total_steps = 200_000

    warmup_steps = 1000
    betas = (0.9, 0.98)
    grad_clip = 1.0

    # Allow CLI overrides
    if args.lr       is not None: peak_lr      = args.lr
    if args.wd       is not None: weight_decay = args.wd
    if args.batch    is not None: batch_size   = args.batch
    if args.ctx      is not None: block_size   = args.ctx
    if args.steps    is not None: total_steps  = args.steps

    # micro_batch: actual per-GPU batch size for each forward pass
    # accum_steps: how many micro batches to accumulate before optimizer step
    # Constraint: micro_batch × accum_steps × world_size == batch_size
    micro_batch = args.micro_batch  # set by CLI, default computed below

    # ---- Device setup ----------------------------------------------------
    use_ddp = int(os.environ.get("RANK", -1)) != -1
    if use_ddp:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = f"cuda:{rank}"
        torch.cuda.set_device(device)
        master     = rank == 0
    else:
        rank       = 0
        world_size = 1
        device     = args.device
        master     = True

    torch.manual_seed(args.seed + rank)

    # bf16 is numerically more stable than fp16 and avoids GradScaler overhead.
    # A800 has native bf16 support (same throughput as fp16).
    is_cuda = "cuda" in device
    if is_cuda and args.bf16:
        dtype = torch.bfloat16
        use_scaler = False
    elif is_cuda:
        dtype = torch.float16
        use_scaler = True
    else:
        dtype = torch.float32
        use_scaler = False

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    ctx = torch.autocast(device_type="cuda", dtype=dtype) if is_cuda else nullcontext()

    # Enable TF32 for matmul and cuDNN (free throughput on Ampere+)
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- Model -----------------------------------------------------------
    cfg = MODEL_CONFIGS.get(args.model)
    if cfg is None:
        raise ValueError(f"Unknown model '{args.model}'. "
                         f"Available: {list(MODEL_CONFIGS.keys())}")
    cfg.block_size = block_size
    model = GPT(cfg).to(device)

    # torch.compile: fuses kernels, eliminates Python overhead (~20-30% speedup).
    # Requires PyTorch >= 2.0.  First step is slow (compilation), then fast.
    if args.compile and is_cuda:
        if master:
            print("Compiling model with torch.compile (first step will be slow)...")
        model = torch.compile(model)

    if use_ddp:
        model = DDP(model, device_ids=[rank])
    raw_model = model.module if use_ddp else model

    # ---- Gradient accumulation setup ------------------------------------
    # Infer micro_batch and accum_steps so that their product equals
    # the paper's global batch size per GPU.
    if micro_batch is None:
        # Default: fit ~4 GB peak activation memory per GPU
        # GPT2-12-12 with ctx=768: ~2GB at micro_batch=32 fp16
        micro_batch = 32
    per_gpu_batch = batch_size // world_size          # target per-GPU tokens
    assert per_gpu_batch >= micro_batch, (
        f"micro_batch ({micro_batch}) > per_gpu_batch ({per_gpu_batch}). "
        f"Reduce --micro_batch or increase --batch."
    )
    accum_steps = per_gpu_batch // micro_batch        # integer division
    # Adjust batch_size to the nearest exact multiple
    effective_batch = micro_batch * accum_steps * world_size

    if master:
        n_params = raw_model.num_parameters()
        print(f"Model: {args.model}  |  params: {n_params/1e6:.1f}M")
        print(f"Dataset  : iGSM-{args.dataset}  |  block_size: {block_size}  |  steps: {total_steps}")
        print(f"Batch    : global={effective_batch}  micro={micro_batch}  "
              f"accum={accum_steps}  world_size={world_size}")

    # ---- Optimizer -------------------------------------------------------
    # Separate weight decay: apply to 2-D tensors (weight matrices) only
    decay_params   = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in raw_model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=peak_lr, betas=betas,
    )

    # ---- Data ------------------------------------------------------------
    # Prefer pre-generated binary file; fall back to on-the-fly generation
    bin_path = os.path.join(args.data_dir,
                            f"igsm_{args.dataset}_train.bin")
    if os.path.exists(bin_path):
        if master:
            print(f"Loading dataset from {bin_path}")
        dataset = TokenBinDataset(bin_path, block_size)
        sampler = (torch.utils.data.DistributedSampler(dataset, shuffle=True)
                   if use_ddp else None)
        loader = DataLoader(
            dataset, batch_size=micro_batch,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4, pin_memory=True,
            prefetch_factor=4,     # prefetch 4 batches per worker
            persistent_workers=True,
        )
        data_iter = iter(loader)
        _epoch = [0]

        def get_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                if use_ddp:
                    _epoch[0] += 1
                    sampler.set_epoch(_epoch[0])
                data_iter = iter(loader)
                return next(data_iter)
    else:
        if master:
            print(f"Binary file not found at {bin_path}; "
                  f"generating on-the-fly (slower).")
        otf = OnTheFlyDataset(args.dataset, block_size,
                              steps_per_epoch=total_steps * accum_steps,
                              batch_size=micro_batch,
                              p_format=args.p_format)
        loader = DataLoader(otf, batch_size=micro_batch,
                            num_workers=4, pin_memory=True)
        data_iter = iter(loader)

        def get_batch():
            nonlocal data_iter
            try:
                return next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                return next(data_iter)

    # ---- Checkpoint resume -----------------------------------------------
    start_step = 0
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir,
                             f"{args.model}_{args.dataset}_latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        if master:
            print(f"Resumed from step {start_step}")

    # ---- Training loop ---------------------------------------------------
    model.train()
    t0 = time.time()

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        desc="Training",
        unit="step",
        disable=not master,     # only rank-0 shows the bar
        dynamic_ncols=True,
    )

    for step in pbar:
        lr = get_lr(step, warmup_steps, total_steps, peak_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Gradient accumulation: accumulate over accum_steps micro-batches
        accum_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(accum_steps):
            x, y = get_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # In DDP, only sync gradients on the last micro-step
            if use_ddp:
                sync = (micro_step == accum_steps - 1)
                ctx_ddp = model.no_sync() if not sync else nullcontext()
            else:
                ctx_ddp = nullcontext()

            with ctx_ddp:
                with ctx:
                    _, loss = model(x, y)
                # Scale loss by accum_steps so gradient magnitude stays constant
                loss = loss / accum_steps
                scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ---- Logging -----------------------------------------------------
        if master and step % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tok_per_sec = args.log_interval * effective_batch * block_size / dt
            pbar.set_postfix({
                "loss": f"{accum_loss:.4f}",
                "lr":   f"{lr:.2e}",
                "tok/s": f"{tok_per_sec/1e3:.1f}k",
            })

        # ---- Checkpointing -----------------------------------------------
        if master and (step % args.save_interval == 0 or step == total_steps - 1):
            ckpt = {
                "model":     raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config":    cfg,
                "step":      step,
                "args":      vars(args),
            }
            torch.save(ckpt, ckpt_path)
            # Also save milestone checkpoints for probing experiments
            if step in {10_000, 25_000, 50_000, 75_000, 100_000,
                        150_000, 200_000}:
                milestone = os.path.join(
                    args.out_dir,
                    f"{args.model}_{args.dataset}_step{step}.pt")
                torch.save(ckpt, milestone)
                print(f"  Saved milestone checkpoint: {milestone}")

    if use_ddp:
        dist.destroy_process_group()

    if master:
        print("Training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pretrain GPT-2 (RoPE) on iGSM dataset")

    # Dataset
    parser.add_argument("--dataset",   choices=["med", "hard"], default="med",
                        help="iGSM dataset: med (max_op=15) or hard (max_op=21)")
    parser.add_argument("--data_dir",  default="data",
                        help="Directory containing igsm_*.bin files")
    parser.add_argument("--p_format",  choices=["pq", "qp"], default="pq",
                        help="Problem format for on-the-fly generation")

    # Model
    parser.add_argument("--model",     default="GPT2-12-12",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model variant (default: GPT2-12-12 = GPT2-small)")

    # Training (override paper defaults)
    parser.add_argument("--lr",          type=float, default=None,
                        help="Peak learning rate (default: 2e-3 per paper)")
    parser.add_argument("--wd",          type=float, default=None,
                        help="Weight decay (default: 0.05/0.03 for med/hard)")
    parser.add_argument("--batch",       type=int,   default=None,
                        help="Global batch size (default: 512/256 for med/hard)")
    parser.add_argument("--micro_batch", type=int,   default=None,
                        help="Per-GPU micro-batch size for each forward pass. "
                             "Gradient accumulation is applied automatically to "
                             "match the global batch size. "
                             "Default: 32 (safe for A800 80GB with ctx=768).")
    parser.add_argument("--ctx",         type=int,   default=None,
                        help="Context length (default: 768/1024 for med/hard)")
    parser.add_argument("--steps",       type=int,   default=None,
                        help="Total training steps (default: 100k/200k)")

    # Infra
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available()
                                                          else "cpu")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--out_dir",       default="checkpoints")
    parser.add_argument("--resume",        action="store_true")
    parser.add_argument("--log_interval",  type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--bf16",          action="store_true", default=True,
                        help="Use bfloat16 (default: True, recommended for A800). "
                             "No GradScaler needed, numerically more stable than fp16.")
    parser.add_argument("--no_bf16",       dest="bf16", action="store_false",
                        help="Disable bf16, fall back to fp16 with GradScaler.")
    parser.add_argument("--compile",       action="store_true", default=True,
                        help="torch.compile the model (~20-30%% speedup, "
                             "first step takes ~1 min to compile).")
    parser.add_argument("--no_compile",    dest="compile", action="store_false",
                        help="Disable torch.compile.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
