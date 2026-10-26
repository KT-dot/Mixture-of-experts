"""High-resolution assignment plots generated only from metrics tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _curve_plot(
    metrics: pd.DataFrame,
    *,
    y_column: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for optimizer, frame in metrics.groupby("optimizer", sort=False):
        ordered = frame.sort_values("training_tokens_seen")
        axis.plot(
            ordered["training_tokens_seen"],
            ordered[y_column],
            marker="o",
            linewidth=2,
            label=optimizer,
        )
    axis.set_xlabel("Training tokens processed")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Optimizer")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_main_plots(metrics: pd.DataFrame, output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _curve_plot(
        metrics,
        y_column="validation_loss",
        y_label="Validation loss",
        title="Validation Loss vs. Training Tokens",
        path=output / "validation_loss_vs_tokens.png",
    )
    _curve_plot(
        metrics,
        y_column="test_bleu",
        y_label="Test continuation BLEU",
        title="Test BLEU vs. Training Tokens",
        path=output / "test_bleu_vs_tokens.png",
    )


def save_sweep_plot(sweep: pd.DataFrame, path: str | Path) -> None:
    """Validation loss against learning-rate multiplier, one line per optimizer."""
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for optimizer, frame in sweep.groupby("optimizer", sort=False):
        ordered = frame.sort_values("lr_multiplier")
        axis.plot(
            ordered["lr_multiplier"],
            ordered["validation_loss"],
            marker="o",
            linewidth=2,
            label=f"{optimizer} (base lr {ordered['lr'].iloc[0] / ordered['lr_multiplier'].iloc[0]:g})",
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Learning-rate multiplier relative to base lr")
    axis.set_ylabel("Validation loss after sweep budget")
    axis.set_title("Learning-Rate Sweep (short-budget proxy runs)")
    axis.grid(True, alpha=0.3)
    axis.legend(title="Optimizer")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
