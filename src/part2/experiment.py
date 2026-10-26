"""Top-level reproducible orchestration for Part 2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .config import with_overrides
from .data import make_synthetic_bundle, prepare_real_data
from .model import DenseTransformer, ModelConfig, parameter_count
from .optimizers import AdamW, Lion, Muon, NAdamW, SophiaG, build_muon_parameter_groups
from .plotting import save_main_plots, save_sweep_plot
from .trainer import train_optimizer


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def make_optimizer(name: str, model: DenseTransformer, values: dict, tokens_per_batch: int = 1):
    settings = dict(values)
    # Consumed by the training loop (Hessian refresh period), not by the optimizer.
    settings.pop("hessian_interval", None)
    if "betas" in settings:
        settings["betas"] = tuple(settings["betas"])
    if name == "adamw":
        return AdamW(model.parameters(), **settings)
    if name == "nadamw":
        return NAdamW(model.parameters(), **settings)
    if name == "lion":
        return Lion(model.parameters(), **settings)
    if name == "sophia":
        return SophiaG(model.parameters(), tokens_per_batch=tokens_per_batch, **settings)
    if name == "muon":
        muon_lr = settings.pop("lr")
        adamw_lr = settings.pop("adamw_lr")
        adamw_weight_decay = settings.pop("adamw_weight_decay")
        weight_decay = settings["weight_decay"]
        if "adamw_betas" in settings:
            settings["adamw_betas"] = tuple(settings["adamw_betas"])
        groups = build_muon_parameter_groups(
            model.named_parameters(),
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            weight_decay=weight_decay,
            adamw_weight_decay=adamw_weight_decay,
        )
        return Muon(groups, lr=muon_lr, **settings)
    raise ValueError(f"unknown optimizer {name!r}")


def _model_config(config: dict, vocab_size: int) -> ModelConfig:
    values = dict(config["model"])
    values.pop("vocab_size", None)
    values["vocab_size"] = vocab_size
    values["max_sequence_length"] = config["training"]["sequence_length"]
    return ModelConfig.from_mapping(values)


def _write_tables(metrics: list[dict], generations: list[dict], output_directory: Path) -> None:
    metrics_frame = pd.DataFrame(metrics)
    generations_frame = pd.DataFrame(generations)
    metrics_frame.to_csv(output_directory / "metrics.csv", index=False)
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    generations_frame.to_csv(output_directory / "generations.csv", index=False)
    save_main_plots(metrics_frame, output_directory)


def _prepare_data_and_initial_state(config: dict, output: Path, synthetic: bool):
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(Path(config["dataset"]["cache_directory"])))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    seed = config["experiment"]["seed"]
    deterministic = config["experiment"]["deterministic_algorithms"]
    seed_everything(seed, deterministic)
    if synthetic:
        synthetic_config = config["synthetic"]
        data = make_synthetic_bundle(
            vocab_size=synthetic_config["vocab_size"],
            sequence_length=config["training"]["sequence_length"],
            train_blocks=synthetic_config["train_blocks"],
            validation_blocks=synthetic_config["validation_blocks"],
            test_blocks=synthetic_config["test_blocks"],
            seed=seed,
        )
    else:
        data = prepare_real_data(config, output)

    model_config = _model_config(config, data.tokenizer.get_vocab_size())
    if model_config.max_sequence_length != data.train.sequence_length:
        raise AssertionError("model and dataset sequence lengths differ")
    seed_everything(seed, deterministic)
    initial_model = DenseTransformer(model_config)
    initial_state = {
        name: tensor.detach().cpu().clone() for name, tensor in initial_model.state_dict().items()
    }
    model_parameter_count = parameter_count(initial_model)
    initial_hash = state_dict_sha256(initial_state)
    torch.save(initial_state, output / "initial_state.pt")
    del initial_model
    return data, model_config, initial_state, initial_hash, model_parameter_count


def _fresh_model_and_optimizer(
    name: str,
    optimizer_values: dict,
    config: dict,
    model_config: ModelConfig,
    initial_state: dict,
    initial_hash: str,
    device: torch.device,
):
    seed_everything(config["experiment"]["seed"], config["experiment"]["deterministic_algorithms"])
    model = DenseTransformer(model_config)
    model.load_state_dict(initial_state)
    if state_dict_sha256(model.state_dict()) != initial_hash:
        raise AssertionError("optimizer run did not start from the saved initial weights")
    model.to(device)
    optimizer = make_optimizer(
        name,
        model,
        optimizer_values,
        tokens_per_batch=config["training"]["batch_size"] * config["training"]["sequence_length"],
    )
    return model, optimizer


def apply_selected_hyperparameters(config: dict, path: str | Path) -> dict:
    """Overlay the learning rates chosen by ``run_sweep`` onto the configuration."""
    selected = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = {name: {"lr": row["selected_lr"]} for name, row in selected.items()}
    return with_overrides(config, {"optimizers": overrides})


def run_sweep(
    config: dict,
    *,
    synthetic: bool = False,
    selected_optimizers: list[str] | None = None,
    output_directory: str | Path | None = None,
) -> dict:
    """Learning-rate sweep on a short prefix of the fixed training order.

    Each trial trains from the shared initial weights for ``sweep.budget_fraction`` of
    one dataset pass with its own complete warmup+cosine schedule, then reports the
    validation loss. Selection never looks at the test split.
    """
    sweep = config["sweep"]
    output = Path(output_directory or config["output"]["directory"]).expanduser().resolve()
    sweep_output = output / "sweep"
    sweep_output.mkdir(parents=True, exist_ok=True)
    data, model_config, initial_state, initial_hash, _ = _prepare_data_and_initial_state(
        config, output, synthetic
    )
    sequence_length = config["training"]["sequence_length"]
    example_budget = max(1, int(len(data.train) * sweep["budget_fraction"]))
    trial_config = with_overrides(
        config,
        {
            "training": {
                "token_budget": example_budget * sequence_length,
                "evaluation_fractions": [1.0],
                "save_checkpoints": False,
            },
            "evaluation": {"enabled": False},
        },
    )
    device = resolve_device(config["training"]["device"])
    budget_tokens = example_budget * sequence_length
    sweep_path = sweep_output / "sweep.csv"
    rows: list[dict] = []
    if sweep_path.exists():
        # Resume: trials are independent (fresh initial weights, fixed data order), so a
        # finished trial with the same optimizer, lr and budget is reused, not rerun.
        previous = pd.read_csv(sweep_path)
        rows = previous[previous["training_tokens"] == budget_tokens].to_dict("records")

    def run_trial(name: str, multiplier: float) -> dict:
        base_lr = config["optimizers"][name]["lr"]
        lr = base_lr * multiplier
        for row in rows:
            if row["optimizer"] == name and math.isclose(row["lr"], lr, rel_tol=1e-9):
                return row
        values = dict(config["optimizers"][name]) | {"lr": lr}
        model, optimizer = _fresh_model_and_optimizer(
            name, values, trial_config, model_config, initial_state, initial_hash, device
        )
        trial_metrics, _ = train_optimizer(
            model=model,
            optimizer=optimizer,
            optimizer_name=f"{name}_lr{multiplier:g}",
            optimizer_hyperparameters=values,
            data=data,
            config=trial_config,
            device=device,
            output_directory=sweep_output,
            initial_state_sha256=initial_hash,
        )
        final = trial_metrics[-1]
        row = {
            "optimizer": name,
            "lr_multiplier": multiplier,
            "lr": lr,
            "training_tokens": final["training_tokens_seen"],
            "final_interval_training_loss": final["training_loss"],
            "validation_loss": final["validation_loss"],
            "skipped_amp_steps": final["skipped_amp_steps"],
            "wall_clock_seconds": final["wall_clock_seconds"],
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(sweep_path, index=False)
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return row

    def best_trial(name: str, grid: list[float]) -> dict:
        # Only trials on the current grid count, so a resumed sweep replays exactly the
        # same sequence of extension decisions as the original run.
        finite = [
            row
            for row in rows
            if row["optimizer"] == name
            and any(math.isclose(row["lr_multiplier"], value) for value in grid)
            and math.isfinite(row["validation_loss"])
        ]
        if not finite:
            raise RuntimeError(f"every sweep trial for {name} diverged")
        return min(finite, key=lambda row: row["validation_loss"])

    extensions: dict[str, int] = {}
    grids: dict[str, list[float]] = {}
    for name in selected_optimizers or config["experiment"]["optimizers"]:
        grid = sorted(sweep["lr_multipliers"])
        for multiplier in grid:
            run_trial(name, multiplier)
        # A minimum on the grid boundary is not a located optimum: step the grid outward
        # by a factor of 2 until the best trial is interior (bounded by max_extensions).
        extensions[name] = 0
        while extensions[name] < sweep.get("max_extensions", 0):
            best = best_trial(name, grid)["lr_multiplier"]
            if math.isclose(best, grid[-1]):
                grid.append(grid[-1] * 2)
                run_trial(name, grid[-1])
            elif math.isclose(best, grid[0]):
                grid.insert(0, grid[0] / 2)
                run_trial(name, grid[0])
            else:
                break
            extensions[name] += 1
        grids[name] = grid

    rows = [
        row
        for row in rows
        if row["optimizer"] in grids
        and any(math.isclose(row["lr_multiplier"], value) for value in grids[row["optimizer"]])
    ]
    frame = pd.DataFrame(rows)
    selected = {}
    for name, extension_count in extensions.items():
        best = best_trial(name, grids[name])
        multipliers = grids[name]
        selected[name] = {
            "selected_lr": float(best["lr"]),
            "selected_multiplier": float(best["lr_multiplier"]),
            "validation_loss": float(best["validation_loss"]),
            "grid_multipliers": multipliers,
            "grid_extensions": extension_count,
            "at_grid_edge": bool(best["lr_multiplier"] in (multipliers[0], multipliers[-1])),
        }
    (sweep_output / "selected_hyperparameters.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_sweep_plot(frame, sweep_output / "sweep_validation_loss_vs_lr.png")
    return {"trials": rows, "selected": selected}


def run_experiment(
    config: dict,
    *,
    synthetic: bool = False,
    selected_optimizers: list[str] | None = None,
    output_directory: str | Path | None = None,
) -> dict:
    output = Path(output_directory or config["output"]["directory"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed = config["experiment"]["seed"]
    deterministic = config["experiment"]["deterministic_algorithms"]
    data, model_config, initial_state, initial_hash, model_parameter_count = (
        _prepare_data_and_initial_state(config, output, synthetic)
    )
    device = resolve_device(config["training"]["device"])
    names = selected_optimizers or config["experiment"]["optimizers"]
    metrics: list[dict] = []
    generations: list[dict] = []
    routing: list[dict] = []
    for name in names:
        if name not in config["optimizers"]:
            raise ValueError(f"optimizer {name!r} has no configuration")
        optimizer_values = config["optimizers"][name]
        model, optimizer = _fresh_model_and_optimizer(
            name, optimizer_values, config, model_config, initial_state, initial_hash, device
        )
        if name == "muon":
            routing = optimizer.routing_table()
            (output / "muon_parameter_routing.json").write_text(
                json.dumps(routing, indent=2), encoding="utf-8"
            )
        run_metrics, run_generations = train_optimizer(
            model=model,
            optimizer=optimizer,
            optimizer_name=name,
            optimizer_hyperparameters=optimizer_values,
            data=data,
            config=config,
            device=device,
            output_directory=output,
            initial_state_sha256=initial_hash,
        )
        metrics.extend(run_metrics)
        generations.extend(run_generations)
        _write_tables(metrics, generations, output)
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "seed": seed,
        "deterministic_algorithms_requested": deterministic,
        "cuda_determinism_limitation": (
            "Some CUDA kernels or external libraries can remain nondeterministic across "
            "hardware, driver, and PyTorch versions."
        ),
        "initial_state_sha256": initial_hash,
        "model_parameter_count": model_parameter_count,
        "model_configuration": model_config.__dict__,
        "data": data.metadata,
        "training_token_budget": (
            data.train.training_tokens
            if config["training"].get("token_budget") is None
            else config["training"]["token_budget"]
        ),
        "data_order_seed": seed,
        "optimizer_order": names,
        "device": str(device),
        "torch_version": torch.__version__,
        "synthetic": synthetic,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    clean_config = {key: value for key, value in config.items() if not key.startswith("_")}
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(clean_config, sort_keys=False), encoding="utf-8"
    )
    return {"metrics": metrics, "generations": generations, "metadata": metadata}
