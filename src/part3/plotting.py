"""Report figures generated only from the saved Part 3 tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd

# Fixed categorical order (validated palette); each family keeps its colour and marker.
FAMILY_STYLE = {
    "greedy / beam": ("#2a78d6", "o"),
    "top-k": ("#eb6834", "s"),
    "top-p": ("#1baf7a", "^"),
    "pure sampling": ("#eda100", "D"),
}
INK = "#0b0b0b"
MUTED = "#52514e"


def _family(row) -> str:
    if row["type"] in ("greedy", "beam"):
        return "greedy / beam"
    if pd.notna(row.get("top_k")):
        return "top-k"
    if pd.notna(row.get("top_p")):
        return "top-p"
    return "pure sampling"


def _style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(MUTED)
    axis.tick_params(colors=MUTED, labelcolor=INK)
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_axisbelow(True)


def save_metric_bars(summary: pd.DataFrame, path: Path) -> None:
    metrics = [
        ("accuracy", "Token accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    ]
    summary = summary.sort_values("order")
    families = summary.apply(_family, axis=1)
    colours = [FAMILY_STYLE[family][0] for family in families]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (column, label) in zip(axes.flat, metrics, strict=True):
        errors = summary.get(f"{column}_seed_std")
        axis.bar(
            summary["strategy"],
            summary[column],
            color=colours,
            width=0.7,
            edgecolor="white",
            linewidth=2,
            yerr=None if errors is None else errors.fillna(0.0),
            error_kw={"ecolor": INK, "capsize": 3, "linewidth": 1},
        )
        axis.set_title(label, color=INK)
        _style_axis(axis)
        axis.tick_params(axis="x", rotation=60)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_STYLE[name][0]) for name in FAMILY_STYLE]
    figure.legend(handles, FAMILY_STYLE, loc="upper center", ncol=4, frameon=False)
    figure.text(
        0.5,
        0.005,
        "Error bars: std across sampling seeds (deterministic strategies have one run).",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.95))
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_quality_diversity(summary: pd.DataFrame, path: Path) -> None:
    summary = summary.sort_values("order")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    reference = summary["reference_perplexity"].iloc[0]
    families = summary.apply(_family, axis=1)
    # beam_1 is greedy decoding (see greedy_beam1_agreement.json): label the pair once.
    labels = {name: name for name in summary["strategy"]}
    if {"greedy", "beam_1"} <= set(labels):
        labels["greedy"] = "greedy = beam_1"
        labels["beam_1"] = None
    for axis, (column, label) in zip(
        axes, [("f1", "F1 vs reference"), ("distinct_2", "Distinct-2 (diversity)")], strict=True
    ):
        for family, (colour, marker) in FAMILY_STYLE.items():
            rows = summary[families == family]
            if rows.empty:
                continue
            axis.scatter(
                rows["generation_perplexity"],
                rows[column],
                color=colour,
                marker=marker,
                s=70,
                label=family,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            stacked = 0
            for _, row in rows.iterrows():
                text = labels[row["strategy"]]
                if text is None:
                    continue
                # Deterministic strategies nearly coincide in diversity; stack those labels.
                stack = family == "greedy / beam" and column == "distinct_2"
                dy = 4 + 11 * stacked if stack else 4
                stacked += stack
                axis.annotate(
                    text,
                    (row["generation_perplexity"], row[column]),
                    textcoords="offset points",
                    xytext=(6, dy),
                    fontsize=8,
                    color=INK,
                )
        axis.axvline(reference, color=MUTED, linestyle="--", linewidth=1)
        axis.annotate(
            "reference continuation PPL",
            (reference, axis.get_ylim()[0]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=8,
            color=MUTED,
        )
        axis.set_xscale("log")
        ticks = [2, 5, 10, 20, 50, 100]
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(tick) for tick in ticks])
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_xlabel("Perplexity of generated continuation under Pythia-160M (log scale)")
        axis.set_ylabel(label)
        _style_axis(axis)
    axes[0].legend(frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_beam_timing(summary: pd.DataFrame, timing: pd.DataFrame | None, path: Path) -> None:
    beams = summary[summary["type"] == "beam"].sort_values("beam_width")
    panels = 1 + (timing is not None and not timing.empty)
    figure, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 5), squeeze=False)
    axis = axes[0, 0]
    axis.plot(
        beams["beam_width"],
        beams["decode_seconds"] / 60,
        color="#2a78d6",
        marker="o",
        markersize=8,
        linewidth=2,
    )
    for _, row in beams.iterrows():
        axis.annotate(
            f"{row['decode_seconds'] / 60:.1f} min",
            (row["beam_width"], row["decode_seconds"] / 60),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=9,
            color=INK,
        )
    axis.set_xticks(beams["beam_width"])
    axis.set_xlabel("Beam width")
    axis.set_ylabel("Decode time for the full evaluation set (minutes)")
    axis.set_title(f"Batched ({int(beams['stories'].iloc[0])} stories)", color=INK)
    _style_axis(axis)
    if panels == 2:
        axis = axes[0, 1]
        styles = (("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"))
        for (batch_size, rows), (colour, marker) in zip(
            timing.groupby("batch_size"), styles, strict=False
        ):
            rows = rows.sort_values("beam_width")
            axis.plot(
                rows["beam_width"],
                rows["ms_per_forward_call"],
                color=colour,
                marker=marker,
                markersize=8,
                linewidth=2,
                label=f"{batch_size} prompt{'s' if batch_size > 1 else ''} per batch",
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(timing["beam_width"].unique()))
        axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.set_xlabel("Beam width")
        axis.set_ylabel("Milliseconds per decoding step (forward call)")
        axis.set_title(f"Per-step cost ({int(timing['stories'].iloc[0])} stories)", color=INK)
        axis.legend(frameon=False)
        _style_axis(axis)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_plots(output: str | Path) -> None:
    output = Path(output)
    summary_path = output / "summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if summary.empty:
        return
    save_metric_bars(summary, output / "decoding_metrics.png")
    save_quality_diversity(summary, output / "quality_vs_diversity.png")
    if (summary["type"] == "beam").any():
        timing_path = output / "beam_timing.csv"
        timing = pd.read_csv(timing_path) if timing_path.exists() else None
        save_beam_timing(summary, timing, output / "beam_width_timing.png")
