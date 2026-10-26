"""Compact figures for report.pdf, drawn only from the CSVs in report/tables/.

Run from the repository root:  .venv/bin/python report/make_report_figures.py
Writes PDFs to report/figures/report/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
OUT = ROOT / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

TEXT_WIDTH_IN = 6.7  # A4 with 2 cm margins
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#222222", "#666666", "#e4e4e4"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def part1_summary():
    ppl = pd.read_csv(TABLES / "part1" / "part1_test_ppl_and_1000_triplet_bleu.csv")
    bleu = pd.read_csv(TABLES / "part1" / "part1_full_test_bleu.csv")
    models = ppl["model"].to_numpy()
    labels = [f"M{m}" for m in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 1.95))

    ax = axes[0]
    ax.plot(x, ppl["test_ppl"], "o", color=SERIES[0])
    for xi, v in zip(x, ppl["test_ppl"]):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5, color=INK)
    ax.set_title("Test perplexity (lower is better)")
    ax.set_xticks(x, labels)
    ax.set_ylim(5.10, 5.43)

    ax = axes[1]
    ci = bleu["bleu_ci95"].str.split("|", expand=True).astype(float)
    lo, hi = bleu["bleu"] - ci[0], ci[1] - bleu["bleu"]
    ax.errorbar(x, bleu["bleu"], yerr=[lo, hi], fmt="o", color=SERIES[0], ecolor=MUTED, elinewidth=1, capsize=2.5)
    ax.set_title("Full-test BLEU with 95% CI")
    ax.set_xticks(x, labels)

    ax = axes[2]
    base = bleu.iloc[0]
    for j, (col, name) in enumerate([("bleu_vi", "vi→en"), ("bleu_ja", "ja→en")]):
        delta = bleu[col] - base[col]
        ax.plot(x + (j - 0.5) * 0.18, delta, MARKERS[j], color=SERIES[j], label=name)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title("BLEU change vs dense M1")
    ax.set_ylabel("BLEU points")
    ax.set_xticks(x, labels)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.1)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "p1_summary.pdf")
    plt.close(fig)


def part1_routing():
    rates = pd.read_csv(TABLES / "part1" / "part1_expert_selection_rates.csv")
    regions = [("source", "Source tokens"), ("English-prefix", "English-prefix tokens")]
    fig, axes = plt.subplots(4, 2, figsize=(TEXT_WIDTH_IN, 4.5), sharex=True)
    for i, model in enumerate([2, 3, 4, 5]):
        for j, (region, title) in enumerate(regions):
            ax = axes[i, j]
            sub = rates[(rates["model"] == model) & (rates["region"] == region)]
            experts = sorted(sub["expert"].unique())
            cols = [(layer, lang) for layer in range(1, 7) for lang in ("vi", "ja")]
            grid = np.array(
                [
                    [
                        sub[(sub["expert"] == e) & (sub["layer"] == layer) & (sub["language"] == lang)][
                            "selection_rate"
                        ].item()
                        for layer, lang in cols
                    ]
                    for e in experts
                ]
            )
            im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    v = grid[r, c]
                    ax.text(c, r, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.2, color="white" if v > 0.55 else INK)
            ax.grid(False)
            for b in range(1, 6):
                ax.axvline(2 * b - 0.5, color="white", linewidth=1.6)
            names = [f"{e}*" if (model == 4 and e == "E0") else e for e in experts]
            ax.set_yticks(range(len(experts)), names, fontsize=6.5)
            ax.set_xticks(range(12), [lang for _, lang in cols], fontsize=6)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.set_ylabel(f"Model {model}", fontsize=8, color=INK)
            if i == 3:
                sec = ax.secondary_xaxis("bottom")
                sec.set_xticks([2 * b + 0.5 for b in range(6)], [f"layer {b + 1}" for b in range(6)], fontsize=6.5)
                sec.tick_params(length=0, pad=10)
                sec.spines["bottom"].set_visible(False)
    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    fig.align_ylabels(axes[:, 0])
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("Fraction of tokens that select the expert", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    fig.savefig(OUT / "p1_routing.pdf")
    plt.close(fig)


def part2_curves():
    metrics = pd.read_csv(TABLES / "part2" / "metrics.csv")
    sweep = pd.read_csv(TABLES / "part2" / "lr_sweep.csv")
    order = ["adamw", "nadamw", "lion", "muon", "sophia"]
    names = {"adamw": "AdamW", "nadamw": "NAdamW", "lion": "Lion", "muon": "Muon", "sophia": "Sophia"}

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.15))
    for k, opt in enumerate(order):
        sub = metrics[metrics["optimizer"] == opt].sort_values("checkpoint_fraction")
        tokens = sub["training_tokens_seen"] / 1e6
        style = dict(color=SERIES[k], marker=MARKERS[k], label=names[opt])
        axes[0].plot(tokens, sub["validation_loss"], **style)
        axes[1].plot(tokens, sub["test_bleu"], **style)
        sw = sweep[sweep["optimizer"] == opt].sort_values("lr_multiplier")
        axes[2].plot(sw["lr_multiplier"], sw["validation_loss"], **style)
        best = sw.loc[sw["validation_loss"].idxmin()]
        axes[2].plot(best["lr_multiplier"], best["validation_loss"], marker=MARKERS[k], markersize=9,
                     markerfacecolor="none", markeredgecolor=SERIES[k], linestyle="none")

    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Training tokens (millions)")
    axes[1].set_title("Test BLEU (8,510 test blocks)")
    axes[1].set_xlabel("Training tokens (millions)")
    axes[2].set_title("LR sweep (0.1× budget proxy)")
    axes[2].set_xlabel("LR multiplier of base LR")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([0.25, 0.5, 1, 2, 4, 8], ["¼", "½", "1", "2", "4", "8"])
    axes[2].set_ylabel("Validation loss")
    for ax in axes[:2]:
        ax.set_xticks([0, 10, 20, 30, 40])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.07), handlelength=1.8)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / "p2_curves.pdf")
    plt.close(fig)


# (label, dx, dy, ha) in points; placed by hand so no two labels collide.
LABELS_F1 = {
    "greedy": ("greedy = beam_1", 5, 1, "left"),
    "beam_2": ("beam_2", 5, -7, "left"),
    "beam_4": ("beam_4", 5, -3, "left"),
    "top_k_5": ("top_k_5", -4, 3, "right"),
    "top_k_20": ("top_k_20", 3, 4, "left"),
    "top_k_50": ("top_k_50", 5, -3, "left"),
    "top_p_0.5": ("top_p_0.5", 4, -9, "left"),
    "top_p_0.9": ("top_p_0.9", 5, -2, "left"),
    "top_p_0.95": ("top_p_0.95", 5, -2, "left"),
    "pure_sampling": ("pure_sampling", -5, -2, "right"),
}
LABELS_D2 = {
    "greedy": ("greedy, beam_1/2/4", 6, -2, "left"),
    "top_k_5": ("top_k_5", 5, -6, "left"),
    "top_p_0.5": ("top_p_0.5", -4, 1, "right"),
    "top_k_20": ("top_k_20", 5, -3, "left"),
    "top_k_50": ("top_k_50", 5, -3, "left"),
    "top_p_0.9": ("top_p_0.9", 5, -3, "left"),
    "top_p_0.95": ("top_p_0.95", 5, -3, "left"),
    "pure_sampling": ("pure_sampling", -5, -2, "right"),
}


def part3_quality_diversity():
    s = pd.read_csv(TABLES / "part3" / "summary.csv").sort_values("order")
    ref_ppl = s["reference_perplexity"].iloc[0]
    families = [("greedy / beam", ("greedy", "beam")), ("top-k", ("top_k",)), ("top-p", ("top_p",)), ("pure sampling", ("pure",))]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.1))
    for ax, col, ylabel in [(axes[0], "f1", "F1 vs reference"), (axes[1], "distinct_2", "Distinct-2 (diversity)")]:
        for k, (fam, prefixes) in enumerate(families):
            sub = s[s["strategy"].str.startswith(prefixes)]
            ax.plot(sub["generation_perplexity"], sub[col], MARKERS[k], color=SERIES[k], label=fam, linestyle="none")
        offsets = LABELS_F1 if col == "f1" else LABELS_D2
        for _, r in s.iterrows():
            if r["strategy"] not in offsets:
                continue
            text, dx, dy, ha = offsets[r["strategy"]]
            ax.annotate(text, (r["generation_perplexity"], r[col]), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=6, color=INK)
        ax.axvline(ref_ppl, color=MUTED, linestyle="--", linewidth=0.9)
        ax.text(ref_ppl * 1.05, ax.get_ylim()[0], f"human reference PPL {ref_ppl:.1f}", fontsize=6, color=MUTED, va="bottom")
        ax.set_xscale("log")
        ax.set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
        ax.set_xlim(1.6, 140)
        ax.set_xlabel("Perplexity of own output under Pythia-160M (log)")
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0.112, 0.199)
    axes[1].legend(loc="upper left", frameon=False, handletextpad=0.2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / "p3_quality_diversity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    part1_summary()
    part1_routing()
    part2_curves()
    part3_quality_diversity()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
