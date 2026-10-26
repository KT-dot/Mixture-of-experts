"""YAML loading and validation for the Part 3 decoding experiment."""

from __future__ import annotations

from pathlib import Path

import yaml

STRATEGY_TYPES = ("greedy", "beam", "sample")
REQUIRED_BEAM_WIDTHS = (1, 2, 4)


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a mapping")
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    for section in ("experiment", "model", "dataset", "generation", "strategies", "output"):
        if section not in config:
            raise ValueError(f"missing configuration section: {section}")
    names = [strategy["name"] for strategy in config["strategies"]]
    if len(names) != len(set(names)):
        raise ValueError("strategy names must be unique")
    for strategy in config["strategies"]:
        kind = strategy.get("type")
        if kind not in STRATEGY_TYPES:
            raise ValueError(f"strategy {strategy['name']!r} has unknown type {kind!r}")
        if kind == "beam" and int(strategy.get("beam_width", 0)) < 1:
            raise ValueError(f"beam strategy {strategy['name']!r} needs beam_width >= 1")
        if kind == "sample" and not strategy.get("seeds"):
            raise ValueError(f"sampling strategy {strategy['name']!r} needs at least one seed")
    widths = {
        strategy["beam_width"] for strategy in config["strategies"] if strategy["type"] == "beam"
    }
    missing = set(REQUIRED_BEAM_WIDTHS) - widths
    if missing:
        raise ValueError(f"the assignment requires beam widths {sorted(missing)}")
    if config["dataset"].get("min_samples", 1000) < 1000:
        raise ValueError("the assignment requires evaluating at least 1000 samples")
