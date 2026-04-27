# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Data generation script for Phase 1 of iGSM reproduction.
# Generates iGSM-med and iGSM-hard datasets and saves them as binary files
# containing concatenated token IDs (uint16), ready for GPT-2 pretraining.
#
# Token format per sample: [222] + prob_token + [223] + sol_token + [224] + ans_token + [50256]
# Special tokens: 222=<prob>, 223=<sol>, 224=<ans>, 50256=<eos>
#
# Usage:
#   python generate_dataset.py --type med --split train --num_samples 500000 --out_dir data/
#   python generate_dataset.py --type med --split train --num_samples 500000 --out_dir data/ --workers 10
#   python generate_dataset.py --type hard --split train --num_samples 500000 --out_dir data/
#   python generate_dataset.py --type med --split test --num_samples 5000 --out_dir data/

import os
import random
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Queue, cpu_count

from const.params import train_bin, test_bin

# How often each worker reports progress (every N samples)
_REPORT_INTERVAL = 50


def _worker(args):
    """Worker function — generates a shard and reports progress via queue."""
    tpy, split, n_samples, p_format, perm_level, detail_level, seed, progress_queue = args

    import random
    from data_gen.pretrain.id_gen import IdGen
    from tools.tools import fix_seed

    fix_seed(seed)
    max_op   = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28
    ava_hash = train_bin if split == "train" else test_bin

    id_gen = IdGen(max_op=max_op, max_edge=max_edge,
                   perm_level=perm_level, detail_level=detail_level)

    tokens = []
    failed = 0
    for i in range(n_samples):
        try:
            # Sample op using the paper's "light" distribution (min(t0,t1)),
            # which biases toward smaller op values — matching the training
            # distribution described in the paper.
            # Crucially we also set id_gen.op (the constructor-level attribute)
            # so that gen_param_light takes the self.op != None branch
            # (self.s = self.op) rather than re-sampling s = min(t0,t1),
            # which would make high-op samples essentially impossible to generate.
            new_op = id_gen.gen_sol_op("light")
            id_gen.op  = new_op
            id_gen.op_ = new_op
            id_gen.perm_level_  = (random.randint(0, 6)  if id_gen.perm_level  is None
                                   else id_gen.perm_level)
            id_gen.detail_level_ = (random.randint(0, 11) if id_gen.detail_level is None
                                    else id_gen.detail_level)
            id_gen.gen_prob(ava_hash, p_format=p_format)
            tokens.extend(id_gen.token_id)
        except Exception:
            failed += 1
        # Report progress in batches to avoid queue overhead
        if (i + 1) % _REPORT_INTERVAL == 0:
            progress_queue.put(_REPORT_INTERVAL)

    # Report any remaining progress
    remainder = n_samples % _REPORT_INTERVAL
    if remainder:
        progress_queue.put(remainder)

    progress_queue.put(None)  # sentinel: this worker is done
    return tokens, failed


def generate_dataset(
    tpy: str,
    split: str,
    num_samples: int,
    out_dir: str,
    p_format: str = "pq",
    perm_level: int = 5,
    detail_level: int = 0,
    seed: int = 42,
    workers: int = 1,
):
    assert tpy in ("med", "hard"), "tpy must be 'med' or 'hard'"
    assert split in ("train", "test"), "split must be 'train' or 'test'"

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"igsm_{tpy}_{split}.bin")

    if workers == 1:
        # Single-process path (simple, no spawn overhead)
        from data_gen.pretrain.id_gen import IdGen
        from tools.tools import fix_seed

        fix_seed(seed)
        max_op   = 15 if tpy == "med" else 21
        max_edge = 20 if tpy == "med" else 28
        ava_hash = train_bin if split == "train" else test_bin

        id_gen = IdGen(max_op=max_op, max_edge=max_edge,
                       perm_level=perm_level, detail_level=detail_level)
        all_tokens, total_failed = [], 0
        for _ in tqdm(range(num_samples), desc=f"iGSM-{tpy} {split}"):
            try:
                new_op = id_gen.gen_sol_op("light")
                id_gen.op  = new_op
                id_gen.op_ = new_op
                id_gen.perm_level_   = (random.randint(0, 6)  if id_gen.perm_level  is None
                                        else id_gen.perm_level)
                id_gen.detail_level_ = (random.randint(0, 11) if id_gen.detail_level is None
                                        else id_gen.detail_level)
                id_gen.gen_prob(ava_hash, p_format=p_format)
                all_tokens.extend(id_gen.token_id)
            except Exception:
                total_failed += 1
    else:
        # Multi-process path: split work evenly across workers
        from multiprocessing import Manager
        per_worker = num_samples // workers
        remainder  = num_samples % workers

        manager = Manager()
        progress_queue = manager.Queue()

        worker_args = [
            (tpy, split,
             per_worker + (1 if i < remainder else 0),
             p_format, perm_level, detail_level,
             seed + i,           # different seed per worker
             progress_queue)
            for i in range(workers)
        ]

        all_tokens, total_failed = [], 0
        print(f"Spawning {workers} workers for iGSM-{tpy} {split} ({num_samples:,} samples)...")

        with Pool(processes=workers) as pool:
            async_result = pool.map_async(_worker, worker_args)

            # Drive the progress bar from the queue while workers run
            with tqdm(total=num_samples, desc=f"iGSM-{tpy} {split}") as pbar:
                done_workers = 0
                while done_workers < workers:
                    msg = progress_queue.get()
                    if msg is None:
                        done_workers += 1
                    else:
                        pbar.update(msg)

            results = async_result.get()

        for tokens, failed in results:
            all_tokens.extend(tokens)
            total_failed += failed

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(out_path)

    total_tokens = len(all_tokens)
    generated    = num_samples - total_failed
    print(f"\nSaved {out_path}")
    print(f"  Samples generated : {generated:,}  (failed: {total_failed})")
    print(f"  Total tokens      : {total_tokens:,}")
    print(f"  File size         : {arr.nbytes / 1e6:.1f} MB")
    print(f"  Avg tokens/sample : {total_tokens / max(1, generated):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Generate iGSM dataset binary files")
    parser.add_argument("--type",        choices=["med", "hard"], default="med")
    parser.add_argument("--split",       choices=["train", "test"], default="train")
    parser.add_argument("--num_samples", type=int, default=500_000)
    parser.add_argument("--out_dir",     type=str, default="data")
    parser.add_argument("--p_format",    choices=["pq", "qp"], default="pq")
    parser.add_argument("--perm_level",  type=int, default=5)
    parser.add_argument("--detail_level",type=int, default=0)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--workers",     type=int, default=cpu_count(),
                        help=f"Number of parallel workers (default: all CPUs = {cpu_count()})")
    args = parser.parse_args()

    generate_dataset(
        tpy=args.type,
        split=args.split,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
        p_format=args.p_format,
        perm_level=args.perm_level,
        detail_level=args.detail_level,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
