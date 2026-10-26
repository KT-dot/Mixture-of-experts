from __future__ import annotations

from pathlib import Path

from src.part2.config import load_config, with_overrides
from src.part2.experiment import apply_selected_hyperparameters, run_experiment, run_sweep


def test_synthetic_end_to_end_writes_metrics_and_plots(tmp_path: Path) -> None:
    config = load_config("configs/part2.yaml")
    config = with_overrides(
        config,
        {
            "model": {
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "mlp_hidden_size": 32,
                "dropout": 0.0,
            },
            "training": {
                "sequence_length": 6,
                "batch_size": 2,
                "token_budget": 12,
                "evaluation_fractions": [1.0],
                "warmup_fraction": 0.0,
                "minimum_lr_ratio": 1.0,
                "device": "cpu",
                "precision": "float32",
                "save_checkpoints": False,
            },
            "evaluation": {
                "batch_size": 2,
                "max_validation_examples": 2,
                "generation_tokens": 2,
                "bleu_examples": 2,
                "generation_batch_size": 2,
            },
            "synthetic": {
                "vocab_size": 16,
                "train_blocks": 4,
                "validation_blocks": 2,
                "test_blocks": 2,
            },
        },
    )
    result = run_experiment(
        config,
        synthetic=True,
        selected_optimizers=["adamw"],
        output_directory=tmp_path,
    )
    assert len(result["metrics"]) == 1
    assert result["metrics"][0]["training_tokens_seen"] == 12
    for filename in (
        "metrics.csv",
        "metrics.json",
        "validation_loss_vs_tokens.png",
        "test_bleu_vs_tokens.png",
        "generations.csv",
    ):
        path = tmp_path / filename
        assert path.exists() and path.stat().st_size > 0


def test_synthetic_sweep_selects_by_validation_loss(tmp_path: Path) -> None:
    config = with_overrides(
        load_config("configs/part2.yaml"),
        {
            "model": {
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "mlp_hidden_size": 32,
                "dropout": 0.0,
            },
            "training": {
                "sequence_length": 6,
                "batch_size": 2,
                "warmup_fraction": 0.0,
                "device": "cpu",
                "precision": "float32",
            },
            "evaluation": {"batch_size": 2, "max_validation_examples": 2},
            "synthetic": {
                "vocab_size": 16,
                "train_blocks": 8,
                "validation_blocks": 2,
                "test_blocks": 2,
            },
            "sweep": {
                "budget_fraction": 0.5,
                "lr_multipliers": [0.5, 1.0, 2.0],
                "max_extensions": 0,
            },
        },
    )
    result = run_sweep(
        config, synthetic=True, selected_optimizers=["adamw", "sophia"], output_directory=tmp_path
    )
    assert len(result["trials"]) == 6
    assert all(trial["training_tokens"] == 4 * 6 for trial in result["trials"])
    for name in ("adamw", "sophia"):
        trials = [trial for trial in result["trials"] if trial["optimizer"] == name]
        best = min(trials, key=lambda trial: trial["validation_loss"])
        assert result["selected"][name]["selected_lr"] == best["lr"]
    selected_path = tmp_path / "sweep" / "selected_hyperparameters.json"
    assert (tmp_path / "sweep" / "sweep_validation_loss_vs_lr.png").stat().st_size > 0
    updated = apply_selected_hyperparameters(config, selected_path)
    assert updated["optimizers"]["adamw"]["lr"] == result["selected"]["adamw"]["selected_lr"]


def test_sophia_hessian_refresh_runs_inside_training_loop(tmp_path: Path) -> None:
    config = with_overrides(
        load_config("configs/part2.yaml"),
        {
            "model": {
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "mlp_hidden_size": 32,
                "dropout": 0.0,
            },
            "training": {
                "sequence_length": 6,
                "batch_size": 2,
                "token_budget": 48,
                "evaluation_fractions": [0.5, 1.0],
                "warmup_fraction": 0.0,
                "device": "cpu",
                "precision": "float32",
                "save_checkpoints": True,
            },
            "evaluation": {
                "batch_size": 2,
                "max_validation_examples": 2,
                "generation_tokens": 2,
                "bleu_examples": None,
                "generation_batch_size": 2,
            },
            "optimizers": {"sophia": {"hessian_interval": 2}},
            "synthetic": {
                "vocab_size": 16,
                "train_blocks": 8,
                "validation_blocks": 2,
                "test_blocks": 3,
            },
        },
    )
    result = run_experiment(
        config, synthetic=True, selected_optimizers=["sophia"], output_directory=tmp_path
    )
    final = result["metrics"][-1]
    # 4 batches with refresh on batch_count % 2 == 1 -> batches 1 and 3.
    assert final["optimizer_steps"] == 4 and final["skipped_amp_steps"] == 0
    assert final["hessian_updates"] == 2 and final["hessian_updates_rejected"] == 0
    assert final["test_bleu_examples"] == 3
    assert 0.0 <= final["sophia_clipped_coordinate_fraction"] <= 1.0
    checkpoints = sorted((tmp_path / "checkpoints" / "sophia").glob("*.pt"))
    assert [path.name for path in checkpoints] == ["fraction_1.0.pt"]
    assert (tmp_path / "progress_sophia.jsonl").read_text().count("\n") == 2


def test_sweep_extends_edge_minimum_and_reuses_finished_trials(tmp_path: Path) -> None:
    config = with_overrides(
        load_config("configs/part2.yaml"),
        {
            "model": {
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "mlp_hidden_size": 32,
                "dropout": 0.0,
            },
            "training": {
                "sequence_length": 6,
                "batch_size": 2,
                "warmup_fraction": 0.0,
                "device": "cpu",
                "precision": "float32",
            },
            "evaluation": {"batch_size": 2, "max_validation_examples": 2},
            "optimizers": {"adamw": {"lr": 1e-4}},
            "synthetic": {
                "vocab_size": 16,
                "train_blocks": 8,
                "validation_blocks": 2,
                "test_blocks": 2,
            },
            # A tiny base lr on a learnable task: the best trial starts on the upper edge.
            "sweep": {"budget_fraction": 0.5, "lr_multipliers": [1.0, 2.0], "max_extensions": 2},
        },
    )
    first = run_sweep(
        config, synthetic=True, selected_optimizers=["adamw"], output_directory=tmp_path
    )
    chosen = first["selected"]["adamw"]
    assert chosen["grid_extensions"] == 2
    assert chosen["grid_multipliers"] == [1.0, 2.0, 4.0, 8.0]
    assert len(first["trials"]) == 4
    again = run_sweep(
        config, synthetic=True, selected_optimizers=["adamw"], output_directory=tmp_path
    )
    assert again["selected"] == first["selected"]
    assert len(again["trials"]) == 4
