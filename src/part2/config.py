"""YAML loading and validation for the Part 2 experiment."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

# The first four cover the required categories; sophia is the optional Hessian bonus.
REQUIRED_OPTIMIZERS = ("adamw", "nadamw", "lion", "muon", "sophia")


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a mapping")
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    for section in (
        "experiment",
        "model",
        "tokenizer",
        "dataset",
        "training",
        "evaluation",
        "optimizers",
        "output",
    ):
        if section not in config:
            raise ValueError(f"missing configuration section: {section}")
    training = config["training"]
    if training["sequence_length"] < 2 or training["batch_size"] < 1:
        raise ValueError("sequence_length must be >=2 and batch_size must be positive")
    fractions = training["evaluation_fractions"]
    if fractions != sorted(set(fractions)) or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("evaluation_fractions must be unique, increasing, and in (0, 1]")
    if fractions[-1] != 1.0:
        raise ValueError("evaluation_fractions must end at 1.0")
    enabled = config["experiment"]["optimizers"]
    unknown = set(enabled) - set(REQUIRED_OPTIMIZERS)
    if unknown:
        raise ValueError(f"unknown optimizers: {sorted(unknown)}")
    if len(enabled) != len(set(enabled)):
        raise ValueError("optimizer list contains duplicates")
    for name in enabled:
        if name not in config["optimizers"]:
            raise ValueError(f"missing hyperparameters for {name}")


def with_overrides(config: dict, overrides: dict) -> dict:
    """Return a deep copy with recursive mapping overrides."""
    result = copy.deepcopy(config)

    def merge(target: dict, source: dict) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)

    merge(result, overrides)
    validate_config(result)
    return result
