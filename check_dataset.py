"""
Dataset distribution checker for iGSM binary files.
Usage:
    python check_dataset.py                        # check all four files
    python check_dataset.py --file data/igsm_med_train.bin --max_op 15
"""
import os
import argparse
import numpy as np
from collections import Counter
from tools.tools import tokenizer

TOK_PROB = 222   # <prob>
TOK_SOL  = 223   # <sol>
TOK_ANS  = 224   # <ans>
TOK_EOS  = 50256 # <eos>


def check_file(bin_path: str, max_op: int, label: str, scan_tokens: int = 0):
    if not os.path.exists(bin_path):
        print(f"  [MISSING] {bin_path}\n")
        return

    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    file_mb = data.nbytes / 1e6
    n_total_tokens = len(data)

    # scan_tokens=0 means scan the whole file
    if scan_tokens == 0 or scan_tokens >= n_total_tokens:
        tokens = data[:].tolist()
    else:
        # Sample evenly-spaced chunks to avoid start-of-file bias
        # (multiprocessing workers write sequentially, so only reading the
        #  beginning would only see worker-0's output)
        chunk = scan_tokens // 5
        step  = max(1, (n_total_tokens - chunk) // 4)
        tokens = []
        for start in range(0, n_total_tokens - chunk + 1, step):
            tokens.extend(data[start: start + chunk].tolist())
            if len(tokens) >= scan_tokens:
                break
        tokens = tokens[:scan_tokens]

    op_counts   = Counter()
    seq_lengths = []     # total token length per sample (222 ... 50256)
    sol_lengths = []     # solution token length (223 ... 224)
    n_samples   = 0
    bad_samples = 0      # missing 223 or 224

    i = 0
    seq_start = None
    while i < len(tokens):
        t = tokens[i]
        if t == TOK_PROB:                   # start of a new sample
            seq_start = i
        elif t == TOK_SOL and seq_start is not None:
            # scan ahead for <ans>
            j = i + 1
            while j < len(tokens) and tokens[j] != TOK_ANS:
                j += 1
            if j >= len(tokens):
                bad_samples += 1
                seq_start = None
                i += 1
                continue

            # count Define steps in the solution
            sol_text = tokenizer.decode(tokens[i+1:j])
            n_steps  = sol_text.count("Define")
            op_counts[n_steps] += 1

            sol_lengths.append(j - i - 1)   # tokens between <sol> and <ans>

            # scan to <eos>
            k = j + 1
            while k < len(tokens) and tokens[k] != TOK_EOS:
                k += 1
            if k < len(tokens):
                seq_lengths.append(k - seq_start + 1)

            n_samples += 1
            seq_start = None
            i = k
        i += 1

    # ── Report ──────────────────────────────────────────────────────────────
    print(f"{'='*66}")
    print(f"  {label}  →  {bin_path}")
    print(f"{'='*66}")
    print(f"  File size          : {file_mb:.1f} MB")
    print(f"  Total tokens       : {n_total_tokens:,}  (scanned: {len(tokens):,})")
    print(f"  Samples found      : {n_samples:,}  (bad/incomplete: {bad_samples})")
    if seq_lengths:
        print(f"  Seq length (tokens): mean={np.mean(seq_lengths):.0f}  "
              f"min={min(seq_lengths)}  max={max(seq_lengths)}")
    if sol_lengths:
        print(f"  Sol length (tokens): mean={np.mean(sol_lengths):.0f}  "
              f"min={min(sol_lengths)}  max={max(sol_lengths)}")

    print(f"\n  Op distribution (# solution steps → sample count):")
    total = sum(op_counts.values())
    missing_ops = []
    for op in range(1, max_op + 1):
        cnt  = op_counts.get(op, 0)
        pct  = 100 * cnt / total if total else 0
        bar  = "█" * min(int(pct * 1.5), 50)
        flag = ""
        if cnt == 0:
            missing_ops.append(op)
            flag = "  ← MISSING"
        print(f"    op={op:2d}: {cnt:6,}  ({pct:4.1f}%)  {bar}{flag}")

    # OOD ops (model won't have seen these in training)
    ood_ops = sorted(k for k in op_counts if k > max_op)
    if ood_ops:
        print(f"\n  OOD ops (op > {max_op}) found in data: {ood_ops}")
        for op in ood_ops:
            print(f"    op={op}: {op_counts[op]}")

    if missing_ops:
        print(f"\n  ⚠  Missing ops: {missing_ops}  — data may be too small or generation failed")
    else:
        print(f"\n  ✓  All ops 1-{max_op} present")

    # Uniformity check: coefficient of variation across in-dist ops
    counts = [op_counts.get(op, 0) for op in range(1, max_op + 1)]
    if any(counts):
        cv = np.std(counts) / (np.mean(counts) + 1e-9)
        print(f"  CV (lower=more uniform): {cv:.3f}"
              + ("  ✓" if cv < 0.5 else "  ⚠ high variance"))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",    default=None,
                        help="Check a single file instead of all four")
    parser.add_argument("--max_op", type=int, default=15)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--scan_tokens", type=int, default=0,
                        help="Tokens to scan per file (0=full file, default). "
                             "For large training files use e.g. 20000000 with "
                             "evenly-spaced sampling across the file.")
    args = parser.parse_args()

    if args.file:
        check_file(args.file, args.max_op, os.path.basename(args.file),
                   args.scan_tokens)
    else:
        files = [
            ("data/igsm_med_train.bin",  15, "iGSM-med  train"),
            ("data/igsm_med_test.bin",   15, "iGSM-med  test"),
            ("data/igsm_hard_train.bin", 21, "iGSM-hard train"),
            ("data/igsm_hard_test.bin",  21, "iGSM-hard test"),
        ]
        for path, max_op, label in files:
            full = os.path.join(args.data_dir, os.path.basename(path)) \
                   if not os.path.isabs(path) else path
            check_file(full, max_op, label, args.scan_tokens)


if __name__ == "__main__":
    main()
