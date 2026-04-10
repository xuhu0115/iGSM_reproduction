# Plotting script to reproduce Figure 3 and Figure 4 of:
#   "Physics of Language Models: Part 2.1" (Ye et al., 2024)
#
# Reads JSON files produced by evaluate.py and generates publication-quality figures.
#
# Usage:
#   # Figure 3 only (single model, single dataset variant)
#   python plot_results.py --fig3 \
#       --json_pq  results/GPT2-12-12_med_pq.json \
#       --json_qp  results/GPT2-12-12_med_qp.json \
#       --dataset  med --out figures/figure3_med.pdf
#
#   # Figure 4 only
#   python plot_results.py --fig4 \
#       --json_pq  results/GPT2-12-12_med_pq.json \
#       --dataset  med --out figures/figure4_med.pdf
#
#   # Both figures
#   python plot_results.py --fig3 --fig4 \
#       --json_pq  results/GPT2-12-12_med_pq.json \
#       --json_qp  results/GPT2-12-12_med_qp.json \
#       --dataset  med --out figures/figure3_4_med.pdf

import json
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Color scheme (matches the paper's green/yellow heat-map style)
# ---------------------------------------------------------------------------

def accuracy_color(acc: float):
    """Map accuracy 0-1 to a green-ish color for the heat-map cells."""
    cmap = plt.cm.RdYlGn
    return cmap(acc)


# ---------------------------------------------------------------------------
# Load JSON results
# ---------------------------------------------------------------------------

def load_results(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    # Convert string keys back to int
    data["results"] = {int(k): v for k, v in data["results"].items()}
    return data


# ---------------------------------------------------------------------------
# Figure 3: Accuracy heat-map table
# ---------------------------------------------------------------------------

def plot_figure3(data_pq: dict, data_qp: dict, dataset: str, out_path: str):
    """
    Reproduces Figure 3: a 2×2 grid of heat-map tables showing test accuracy
    for iGSM-{med/hard}_{pq/qp} across op values (in-dist + OOD).

    data_pq / data_qp: dicts from load_results() for each p_format.
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 5))

    configs = [
        (data_pq, "pq", "beam1·nosample"),
        (data_pq, "pq", "beam4·dosample"),
        (data_qp, "qp", "beam1·nosample"),
        (data_qp, "qp", "beam4·dosample"),
    ]

    for col, (data, fmt, label) in enumerate(configs):
        indist_ops = data["indist_ops"]
        ood_ops    = data["ood_ops"]
        results    = data["results"]

        # Separate beam results if both are available in data, else use what's there
        ops_to_show = indist_ops + ood_ops

        row_labels  = [f"op={op}" for op in ops_to_show]
        accuracies  = [results.get(op, {}).get("accuracy", float("nan"))
                       for op in ops_to_show]

        ax = axes[col // 2][col % 2]
        ax.set_title(f"iGSM-{dataset}_{fmt}\n{label}", fontsize=9)

        # Draw cells
        n = len(ops_to_show)
        for i, (op, acc) in enumerate(zip(ops_to_show, accuracies)):
            if np.isnan(acc):
                color = (0.9, 0.9, 0.9, 1.0)
                text  = "N/A"
            else:
                color = accuracy_color(acc)
                text  = f"{acc*100:.1f}"
            rect = mpatches.FancyBboxPatch(
                (i, 0), 1, 1,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="white", linewidth=0.5
            )
            ax.add_patch(rect)
            ax.text(i + 0.5, 0.55, text,
                    ha="center", va="center", fontsize=7, fontweight="bold")
            ood_marker = "*" if op in ood_ops else ""
            ax.text(i + 0.5, 0.15, f"op={op}{ood_marker}",
                    ha="center", va="center", fontsize=6)

        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.axis("off")

    # Color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("Accuracy", fontsize=9)

    plt.suptitle(
        f"Figure 3: Test accuracies on iGSM-{dataset} (in-dist + OOD)\n"
        f"* = OOD (op > max training op)",
        fontsize=10, y=1.02
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Figure 3 saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Line plot variant (cleaner for a single model)
# ---------------------------------------------------------------------------

def plot_figure3_lineplot(data_pq: dict, data_qp: dict,
                          dataset: str, out_path: str):
    """
    Alternative: line plot of accuracy vs. op (cleaner to read than heat-map).
    Reproduces the spirit of Figure 3 with shaded in-dist / OOD regions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    for ax, (data, fmt) in zip(axes, [(data_pq, "pq"), (data_qp, "qp")]):
        indist_ops = sorted(data["indist_ops"])
        ood_ops    = sorted(data["ood_ops"])
        results    = data["results"]
        all_ops    = sorted(set(indist_ops + ood_ops))

        ops  = [op for op in all_ops if op in results]
        accs = [results[op]["accuracy"] * 100 for op in ops]

        # Shade in-dist vs OOD
        max_indist = max(indist_ops)
        ax.axvspan(min(ops) - 0.5, max_indist + 0.5,
                   alpha=0.08, color="green", label="in-dist")
        ax.axvspan(max_indist + 0.5, max(ops) + 0.5,
                   alpha=0.08, color="red", label="OOD")

        # beam=1 curve (solid) — data already has single beam
        ax.plot(ops, accs, "o-", color="steelblue",
                linewidth=2, markersize=5, label="beam=1")

        ax.set_xlabel("op (solution operations)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(f"iGSM-{dataset}_{fmt}", fontsize=11)
        ax.set_ylim(-5, 105)
        ax.set_xticks(ops)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        f"Figure 3 (line plot): Test accuracy on iGSM-{dataset}",
        fontsize=12
    )
    fig.tight_layout()
    lp_path = out_path.replace(".pdf", "_lineplot.pdf").replace(".png", "_lineplot.png")
    if lp_path == out_path:
        lp_path = out_path + "_lineplot.pdf"
    fig.savefig(lp_path, bbox_inches="tight", dpi=150)
    print(f"Figure 3 (line plot) saved to {lp_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Unnecessary ops / params per correct solution
# ---------------------------------------------------------------------------

def plot_figure4(data_pq: dict, data_qp: dict, dataset: str, out_path: str):
    """
    Reproduces Figure 4: bar/line plot of average unnecessary operations and
    parameters per correct solution, per op value.
    Paper shows: avg_unnecessary_op ≈ 0 and avg_unnecessary_param ≈ 0
    for in-distribution, confirming 'level-1 reasoning'.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharey="row")

    metrics = [
        ("avg_unnecessary_op",    "Avg unnecessary operations"),
        ("avg_unnecessary_param", "Avg unnecessary parameters"),
    ]

    for row, (metric_key, metric_label) in enumerate(metrics):
        for col, (data, fmt) in enumerate([(data_pq, "pq"), (data_qp, "qp")]):
            ax = axes[row][col]
            indist_ops = sorted(data["indist_ops"])
            ood_ops    = sorted(data["ood_ops"])
            results    = data["results"]
            all_ops    = sorted(set(indist_ops + ood_ops))

            ops    = [op for op in all_ops if op in results]
            values = [results[op].get(metric_key, 0.0) for op in ops]
            n_correct = [results[op]["n_correct"] for op in ops]

            colors = ["steelblue" if op in indist_ops else "tomato" for op in ops]
            bars = ax.bar(range(len(ops)), values, color=colors,
                          edgecolor="white", linewidth=0.5)

            # Annotate bars with n_correct count (small text)
            for i, (bar, nc) in enumerate(zip(bars, n_correct)):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 0.005, f"n={nc}",
                        ha="center", va="bottom", fontsize=5, rotation=90)

            ax.set_xticks(range(len(ops)))
            ax.set_xticklabels([f"op={op}" for op in ops],
                               rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(metric_label, fontsize=8)
            ax.set_title(f"iGSM-{dataset}_{fmt}", fontsize=9)
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.3)

            # Legend
            blue_patch = mpatches.Patch(color="steelblue", label="in-dist")
            red_patch  = mpatches.Patch(color="tomato",    label="OOD")
            ax.legend(handles=[blue_patch, red_patch], fontsize=7)

    plt.suptitle(
        f"Figure 4: Avg unnecessary ops/params per correct solution\n"
        f"(≈0 confirms 'level-1' reasoning skill in iGSM-{dataset})",
        fontsize=10
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Figure 4 saved to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 3 & 4 from iGSM paper")

    parser.add_argument("--fig3",     action="store_true", help="Plot Figure 3")
    parser.add_argument("--fig4",     action="store_true", help="Plot Figure 4")
    parser.add_argument("--json_pq",  required=True,
                        help="JSON result file for p_format=pq (from evaluate.py)")
    parser.add_argument("--json_qp",  default=None,
                        help="JSON result file for p_format=qp (optional, "
                             "uses pq results if not provided)")
    parser.add_argument("--dataset",  choices=["med", "hard"], default="med")
    parser.add_argument("--out",      default="figures/figure.pdf",
                        help="Output figure path (.pdf or .png)")

    args = parser.parse_args()

    if not args.fig3 and not args.fig4:
        print("Specify at least one of --fig3 or --fig4")
        return

    data_pq = load_results(args.json_pq)
    data_qp = load_results(args.json_qp) if args.json_qp else data_pq

    stem, ext = os.path.splitext(args.out)
    if not ext:
        ext = ".pdf"

    if args.fig3:
        # Heat-map version
        plot_figure3(data_pq, data_qp, args.dataset, stem + "_fig3_heatmap" + ext)
        # Line plot version
        plot_figure3_lineplot(data_pq, data_qp, args.dataset,
                              stem + "_fig3" + ext)

    if args.fig4:
        plot_figure4(data_pq, data_qp, args.dataset, stem + "_fig4" + ext)


if __name__ == "__main__":
    main()
