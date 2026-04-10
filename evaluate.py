# Evaluation script for iGSM reproduction (Phase 3)
# Reproduces Figure 3 and Figure 4 of:
#   "Physics of Language Models: Part 2.1" (Ye et al., 2024)
#
# Figure 3: Test accuracy per op value (in-dist + OOD), beam=1/4
# Figure 4: Avg unnecessary ops/params per correct solution
#
# Usage:
#   # Evaluate a single checkpoint (beam=1, greedy)
#   python evaluate.py --ckpt checkpoints/GPT2-12-12_med_latest.pt \
#                      --dataset med --n_problems 4096
#
#   # Full Figure 3/4 evaluation (beam=1 and beam=4)
#   python evaluate.py --ckpt checkpoints/GPT2-12-12_med_latest.pt \
#                      --dataset med --n_problems 4096 \
#                      --beam 4 --dosample \
#                      --save_json results/GPT2-12-12_med.json

import os
import sys
import json
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm

from model import GPT, GPTConfig, MODEL_CONFIGS
from tools.tools import tokenizer, fix_seed
from tools.tools_test import true_correct, output_split
from data_gen.pretrain.id_gen import IdGen
from data_gen.prototype.id_gen import IdGen_PT
from const.params import test_bin, mod

# ---------------------------------------------------------------------------
# Special token ids (defined in iGSM)
#   222 = <prob>   223 = <sol>   224 = <ans>   50256 = <eos>
# ---------------------------------------------------------------------------
TOK_PROB = 222
TOK_SOL  = 223
TOK_ANS  = 224
TOK_EOS  = 50256


# ---------------------------------------------------------------------------
# Generate test problems for a fixed op count
# ---------------------------------------------------------------------------

def generate_test_problems(dataset_type: str, op: int,
                            n_problems: int, p_format: str = "pq"):
    """
    Generate `n_problems` test problems with exactly `op` solution operations.
    Returns a list of IdGen objects (each holds .prob_id, .problem, .op_).
    """
    max_op   = 15 if dataset_type == "med" else 21
    max_edge = 20 if dataset_type == "med" else 28

    id_gen = IdGen(max_op=max_op, max_edge=max_edge,
                   perm_level=5, detail_level=0, op=op)
    problems = []
    attempts = 0
    max_attempts = n_problems * 200

    while len(problems) < n_problems and attempts < max_attempts:
        attempts += 1
        try:
            id_gen.gen_prob(test_bin, p_format=p_format)
            # Clone the relevant state we need later
            entry = {
                "prob_id":  list(id_gen.prob_id),     # [50256, 222] + prob_token + [223]
                "token_id": list(id_gen.token_id),    # full reference token sequence
                "problem":  id_gen.problem,           # Problem object for true_correct()
                "op":       id_gen.op_,
            }
            problems.append(entry)
        except Exception:
            continue

    if len(problems) < n_problems:
        print(f"  Warning: only generated {len(problems)}/{n_problems} "
              f"problems for op={op}")
    return problems


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_generate(model: GPT, prompt_ids: list[list[int]],
                   max_new_tokens: int, device: str,
                   beam: int = 1, dosample: bool = False,
                   temperature: float = 1.0) -> list[list[int]]:
    """
    Generate completions for a batch of prompts.

    beam=1: per-sequence greedy/sampling with KV cache (no padding — avoids
            wrong RoPE positions that would arise from left-padding).
    beam>1: per-sequence beam search with KV cache.

    Returns a list of full token sequences (prompt + generated tokens).
    """
    model.eval()
    cfg = model.config
    B = len(prompt_ids)
    generated = [None] * B

    if beam == 1:
        for i in range(B):
            generated[i] = _greedy_generate(
                model, prompt_ids[i], max_new_tokens, device,
                dosample, temperature, cfg.block_size,
            )
    else:
        for i in range(B):
            generated[i] = _beam_sample(
                model, prompt_ids[i], max_new_tokens, device,
                beam, temperature, cfg.block_size,
            )

    return generated


def _greedy_generate(model: GPT, prompt: list[int], max_new_tokens: int,
                     device: str, dosample: bool, temperature: float,
                     block_size: int) -> list[int]:
    """
    Single-sequence greedy / temperature-sampling generation with KV cache.
    No padding — RoPE positions match training exactly.
    """
    prompt_t = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    logits, _, past_kvs = model(prompt_t, use_cache=True, start_pos=0)

    generated = list(prompt)
    cur_len = len(prompt)

    for _ in range(max_new_tokens):
        logits_last = logits[0, -1, :]
        if dosample:
            logits_last = logits_last / temperature
            probs = torch.softmax(logits_last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = logits_last.argmax().item()

        generated.append(next_id)

        if TOK_ANS in generated and TOK_EOS in generated[generated.index(TOK_ANS):]:
            break
        if cur_len >= block_size:
            break

        next_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, _, past_kvs = model(
            next_t, past_kvs=past_kvs, start_pos=cur_len, use_cache=True
        )
        cur_len += 1

    return generated


def _beam_sample(model: GPT, prompt: list[int], max_new_tokens: int,
                 device: str, num_beams: int, temperature: float,
                 block_size: int) -> list[int]:
    """
    Beam search with KV cache for a single prompt.
    Each beam carries its own past_kvs so decode steps are O(1) per token.
    Returns the best (highest log-prob) complete sequence.
    """
    # --- Prefill: process full prompt once, get initial KV cache ---
    prompt_t = torch.tensor(prompt, dtype=torch.long, device=device).unsqueeze(0)
    logits, _, init_kvs = model(prompt_t, use_cache=True, start_pos=0)

    # Sample top-k tokens from the prefill logits to seed the beams
    logits_last = logits[0, -1, :] / temperature
    log_probs_dist = torch.log_softmax(logits_last, dim=-1)
    topk_lp, topk_idx = torch.topk(log_probs_dist, num_beams)

    # Each beam: (cumulative_log_prob, token_list, past_kvs_after_last_token)
    # init_kvs covers positions 0..len(prompt)-1; the first sampled token
    # will be fed at start_pos=len(prompt).
    beams = [
        (lp.item(), list(prompt) + [idx.item()], init_kvs)
        for lp, idx in zip(topk_lp, topk_idx)
    ]
    completed = []
    # cur_len = position of the NEXT token to generate
    cur_len = len(prompt)   # first sampled token is already appended above

    for _ in range(max_new_tokens - 1):
        if not beams:
            break
        all_candidates = []

        for log_prob, seq, beam_kvs in beams:
            # Feed only the last token (the one just appended) with KV cache
            last_t = torch.tensor([[seq[-1]]], dtype=torch.long, device=device)
            logits_step, _, new_kvs = model(
                last_t, past_kvs=beam_kvs, start_pos=cur_len, use_cache=True
            )
            logits_step = logits_step[0, -1, :] / temperature
            lp_dist = torch.log_softmax(logits_step, dim=-1)
            topk_lp2, topk_idx2 = torch.topk(lp_dist, num_beams)

            for lp2, idx2 in zip(topk_lp2.tolist(), topk_idx2.tolist()):
                all_candidates.append((log_prob + lp2, seq + [idx2], new_kvs))

        cur_len += 1
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = []
        for lp, seq, kvs in all_candidates[:num_beams]:
            if TOK_ANS in seq and TOK_EOS in seq[seq.index(TOK_ANS):]:
                completed.append((lp, seq))
            else:
                beams.append((lp, seq, kvs))
        if len(completed) >= num_beams:
            break

    if completed:
        completed.sort(key=lambda x: x[0], reverse=True)
        return completed[0][1]
    elif beams:
        beams.sort(key=lambda x: x[0], reverse=True)
        return beams[0][1]
    return prompt


# ---------------------------------------------------------------------------
# Correctness + redundancy check
# ---------------------------------------------------------------------------

def check_output(generated_tokens: list[int], problem_entry: dict) -> dict:
    """
    Run true_correct() and compute Figure 4 statistics.
    Returns a dict with fields:
        correct (bool), n_op (int), n_unnecessary_op (int),
        n_unnecessary_param (int), n_unnecessary_op_reask (int), ...
    """
    problem = problem_entry["problem"]
    result = {
        "correct": False,
        "n_op": problem_entry["op"],
        "n_unnecessary_op": 0,
        "n_unnecessary_param": 0,
        "n_unnecessary_op_reask": 0,
        "n_unnecessary_param_reask": 0,
    }

    try:
        correct, my_print, parser = true_correct(generated_tokens, problem=problem)
        result["correct"] = correct
        if correct and parser is not None:
            # Figure 4: count unnecessary ops / params in the solution
            gpt_params = set(parser.param_dict.keys())
            nece_params = set(
                problem.get_ntn(param=p) for p in problem.topological_order
            )
            result["n_unnecessary_param"] = len(gpt_params - nece_params)
            result["n_unnecessary_op"] = max(
                0, parser.sol_op - problem.n_op
            )
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Evaluate over a set of op values
# ---------------------------------------------------------------------------

def evaluate_op_range(model: GPT, dataset_type: str, op_values: list[int],
                      n_problems: int, device: str, p_format: str,
                      beam: int, dosample: bool,
                      max_new_tokens: int = 512,
                      batch_size: int = 32) -> dict:
    """
    Evaluate model accuracy for each op value in op_values.
    Returns nested dict: results[op] = {correct, total, avg_unnecessary_op, ...}
    """
    results = {}

    for op in op_values:
        print(f"  Evaluating op={op} ...")
        problems = generate_test_problems(dataset_type, op, n_problems, p_format)
        if not problems:
            continue

        op_results = []
        for start in tqdm(range(0, len(problems), batch_size),
                          desc=f"    op={op}", leave=False):
            batch = problems[start: start + batch_size]
            prompts = [p["prob_id"] for p in batch]  # each: [50256, 222] + prob_tokens + [223]
            generated = batch_generate(
                model, prompts, max_new_tokens, device,
                beam=beam, dosample=dosample,
            )
            for gen_seq, prob_entry in zip(generated, batch):
                r = check_output(gen_seq, prob_entry)
                op_results.append(r)

        n_correct = sum(r["correct"] for r in op_results)
        n_total   = len(op_results)
        correct_results = [r for r in op_results if r["correct"]]

        results[op] = {
            "accuracy":                   n_correct / n_total if n_total > 0 else 0.0,
            "n_correct":                  n_correct,
            "n_total":                    n_total,
            "avg_unnecessary_op":         (sum(r["n_unnecessary_op"]    for r in correct_results)
                                           / len(correct_results)) if correct_results else 0.0,
            "avg_unnecessary_param":      (sum(r["n_unnecessary_param"] for r in correct_results)
                                           / len(correct_results)) if correct_results else 0.0,
        }
        acc_pct = results[op]["accuracy"] * 100
        print(f"    op={op:2d}: accuracy={acc_pct:.1f}%  "
              f"({n_correct}/{n_total})  "
              f"avg_unnecessary_op={results[op]['avg_unnecessary_op']:.3f}  "
              f"avg_unnecessary_param={results[op]['avg_unnecessary_param']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-2 on iGSM and reproduce Figure 3 & 4")

    parser.add_argument("--ckpt",        required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--dataset",     choices=["med", "hard"], default="med",
                        help="Dataset type: med or hard")
    parser.add_argument("--p_format",    choices=["pq", "qp"], default="pq",
                        help="Problem format (pq or qp)")
    parser.add_argument("--n_problems",  type=int, default=4096,
                        help="Problems per op value (paper uses 4096)")
    parser.add_argument("--beam",        type=int, default=1,
                        help="Beam size: 1 (greedy) or 4 (paper)")
    parser.add_argument("--dosample",    action="store_true",
                        help="Multinomial sampling (use with --beam 4)")
    parser.add_argument("--batch_size",  type=int, default=32,
                        help="Inference batch size")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per problem")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available()
                                                        else "cpu")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--save_json",   default=None,
                        help="Save results to this JSON file")

    args = parser.parse_args()
    fix_seed(args.seed)

    # ---- Load model -------------------------------------------------------
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg  = ckpt["config"]
    model = GPT(cfg).to(args.device)
    state_dict = ckpt["model"]
    # torch.compile wraps the model as _orig_mod; strip the prefix if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model: {cfg.n_layer}L-{cfg.n_head}H-{cfg.n_embd}D  "
          f"({model.num_parameters()/1e6:.1f}M params)")

    # ---- Op ranges (matches paper Figure 3) -------------------------------
    if args.dataset == "med":
        # in-distribution: op ∈ {2..15},  OOD: op ∈ {20,21,22,23}
        indist_ops = list(range(2, 16))
        ood_ops    = [20, 21, 22, 23]
        # For "reask" we'd regenerate the query; here we use standard eval
    else:  # hard
        # in-distribution: op ∈ {2..21},  OOD: op ∈ {28,29,30,31,32}
        indist_ops = list(range(2, 22))
        ood_ops    = [28, 29, 30, 31, 32]

    all_ops = indist_ops + ood_ops

    print(f"\n=== Evaluating iGSM-{args.dataset}_{args.p_format} "
          f"(beam={args.beam}, dosample={args.dosample}) ===")
    print(f"In-dist ops : {indist_ops}")
    print(f"OOD ops     : {ood_ops}")

    results = evaluate_op_range(
        model, args.dataset, all_ops,
        n_problems=args.n_problems,
        device=args.device,
        p_format=args.p_format,
        beam=args.beam,
        dosample=args.dosample,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # ---- Summary ----------------------------------------------------------
    print("\n=== Summary (Figure 3) ===")
    print(f"{'op':>4}  {'accuracy':>9}  {'unnecessary_op':>14}  {'unnecessary_param':>17}")
    print("-" * 52)
    for op in all_ops:
        if op not in results:
            continue
        r = results[op]
        marker = " [OOD]" if op in ood_ops else ""
        print(f"{op:>4}  {r['accuracy']*100:>8.1f}%  "
              f"{r['avg_unnecessary_op']:>14.3f}  "
              f"{r['avg_unnecessary_param']:>17.3f}{marker}")

    avg_indist = sum(results[op]["accuracy"] for op in indist_ops if op in results)
    avg_indist /= max(1, len([op for op in indist_ops if op in results]))
    avg_ood    = sum(results[op]["accuracy"] for op in ood_ops if op in results)
    avg_ood   /= max(1, len([op for op in ood_ops if op in results]))
    print(f"\nAvg in-dist accuracy : {avg_indist*100:.1f}%")
    print(f"Avg OOD accuracy     : {avg_ood*100:.1f}%")

    # ---- Save results -----------------------------------------------------
    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        output = {
            "checkpoint": args.ckpt,
            "dataset":    args.dataset,
            "p_format":   args.p_format,
            "beam":       args.beam,
            "dosample":   args.dosample,
            "n_problems": args.n_problems,
            "indist_ops": indist_ops,
            "ood_ops":    ood_ops,
            "results":    {str(k): v for k, v in results.items()},
        }
        with open(args.save_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
