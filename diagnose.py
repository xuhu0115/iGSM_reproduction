"""
Quick diagnostic: load checkpoint, generate one solution, print everything.
Run from the iGSM repo root:
    python diagnose.py [--ckpt PATH] [--op 5] [--p_format pq]
"""
import argparse, torch
from model import GPT
from tools.tools import tokenizer
from data_gen.pretrain.id_gen import IdGen
from const.params import test_bin
from tools.tools_test import true_correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default="checkpoints/GPT2-12-12_med_latest.pt")
    parser.add_argument("--op",       type=int, default=5)
    parser.add_argument("--p_format", default="pq")
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    # ── 1. Checkpoint info ────────────────────────────────────────────────
    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    step = ckpt.get("step", "N/A")
    training_args = ckpt.get("args", {})
    print(f"  step            : {step}")
    print(f"  training args   : {training_args}")

    # ── 2. Load model ─────────────────────────────────────────────────────
    cfg = ckpt["config"]
    model = GPT(cfg).to(args.device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"  model           : {cfg.n_layer}L-{cfg.n_head}H-{cfg.n_embd}D")

    # ── 3. Generate one test problem ──────────────────────────────────────
    id_gen = IdGen(max_op=15, max_edge=20, perm_level=5, detail_level=0, op=args.op)
    for attempt in range(500):
        try:
            id_gen.gen_prob(test_bin, p_format=args.p_format)
            break
        except Exception:
            continue
    else:
        print("Could not generate a test problem after 500 attempts.")
        return

    prompt = list(id_gen.prob_id)   # [50256, 222] + prob_tokens + [223]
    print(f"\nPrompt ({len(prompt)} tokens):\n{tokenizer.decode(prompt)}\n")
    print(f"Reference solution:\n{' '.join(id_gen.problem.solution)}\n")
    print(f"Reference answer  : {id_gen.problem.ans}")

    # ── 4. Autoregressive generation with KV cache ────────────────────────
    prompt_t = torch.tensor(prompt, dtype=torch.long, device=args.device).unsqueeze(0)
    with torch.no_grad():
        logits, _, past_kvs = model(prompt_t, use_cache=True, start_pos=0)

    generated = list(prompt)
    cur_len = len(prompt)
    TOK_ANS = 224
    for _ in range(512):
        next_id = logits[0, -1, :].argmax().item()
        generated.append(next_id)
        if next_id == 50256:
            break
        if cur_len >= cfg.block_size:
            break
        next_t = torch.tensor([[next_id]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            logits, _, past_kvs = model(
                next_t, past_kvs=past_kvs, start_pos=cur_len, use_cache=True
            )
        cur_len += 1

    # ── 5. Decode and display ─────────────────────────────────────────────
    SEP = "-" * 60
    if 223 in generated:
        sol_start = generated.index(223) + 1
        sol_tokens = generated[sol_start:]
        print(f"Model output ({len(sol_tokens)} tokens after <sol>):\n{SEP}")
        print(tokenizer.decode(sol_tokens))
        print(SEP)
    else:
        print("WARNING: <sol> token (223) not found in output!")
        print(tokenizer.decode(generated))

    # ── 6. true_correct check ─────────────────────────────────────────────
    try:
        correct, my_print, parser_obj = true_correct(generated, problem=id_gen.problem)
        print(f"\ntrue_correct → {correct}")
    except Exception as e:
        print(f"\ntrue_correct raised: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
