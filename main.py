"""Repository command-line entry point.

Part 1 (MoE translation) is notebook-driven: see ``src/part1/`` and the README.
Part 2 (optimizers) and Part 3 (decoding strategies) run from here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PART2_OPTIMIZERS = ("adamw", "nadamw", "lion", "muon", "sophia")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced NLP Assignment 2 experiments")
    parser.add_argument("--part", type=int, choices=(2, 3), required=True)
    parser.add_argument("--config", default=None, help="defaults to configs/part<N>.yaml")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="part 2: tiny synthetic token IDs; never accesses Hugging Face",
    )
    parser.add_argument("--optimizers", nargs="+", choices=PART2_OPTIMIZERS, default=None)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="part 2: run the short-budget learning-rate sweep instead of the full run",
    )
    parser.add_argument(
        "--hyperparameters",
        default=None,
        help="part 2: selected_hyperparameters.json written by --sweep",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="part 3: subset of configured strategy names to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="part 3: evaluate only the first N test stories (debugging only)",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def run_part2(arguments: argparse.Namespace) -> None:
    from src.part2.config import load_config, with_overrides
    from src.part2.experiment import apply_selected_hyperparameters, run_experiment, run_sweep

    config = load_config(arguments.config or "configs/part2.yaml")
    if arguments.synthetic_smoke:
        config = with_overrides(
            config,
            {
                "model": {
                    "d_model": 32,
                    "n_heads": 4,
                    "n_layers": 1,
                    "mlp_hidden_size": 64,
                    "dropout": 0.0,
                },
                "training": {
                    "sequence_length": 8,
                    "batch_size": 4,
                    "token_budget": 32,
                    "evaluation_fractions": [1.0],
                    "warmup_fraction": 0.0,
                    "minimum_lr_ratio": 1.0,
                    "device": "cpu",
                    "precision": "float32",
                    "save_checkpoints": False,
                },
                "evaluation": {
                    "batch_size": 4,
                    "max_validation_examples": 4,
                    "generation_tokens": 2,
                    "bleu_examples": 2,
                    "generation_batch_size": 2,
                },
                "sweep": {"budget_fraction": 0.5, "lr_multipliers": [0.5, 1.0]},
            },
        )
    if arguments.hyperparameters:
        config = apply_selected_hyperparameters(config, arguments.hyperparameters)
    output = arguments.output
    if output is None and arguments.synthetic_smoke:
        output = "outputs/part2/synthetic_smoke"
    runner = run_sweep if arguments.sweep else run_experiment
    result = runner(
        config,
        synthetic=arguments.synthetic_smoke,
        selected_optimizers=arguments.optimizers,
        output_directory=output,
    )
    destination = Path(output or config["output"]["directory"]).resolve()
    if arguments.sweep:
        print(f"Completed {len(result['trials'])} sweep trials in {destination / 'sweep'}")
        print(f"Selected: {result['selected']}")
    else:
        print(f"Completed {len(result['metrics'])} evaluation rows in {destination}")


def run_part3(arguments: argparse.Namespace) -> None:
    from src.part3.config import load_config
    from src.part3.experiment import run_experiment

    config = load_config(arguments.config or "configs/part3.yaml")
    result = run_experiment(
        config,
        strategy_names=arguments.strategies,
        limit=arguments.limit,
        output_directory=arguments.output,
    )
    print(f"Completed {len(result['metrics'])} strategy rows in {result['output_directory']}")


def main() -> None:
    arguments = parse_arguments()
    if arguments.part == 2:
        run_part2(arguments)
    else:
        run_part3(arguments)


if __name__ == "__main__":
    main()
