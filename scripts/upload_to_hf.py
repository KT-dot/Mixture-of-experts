"""Publish trained checkpoints to the Hugging Face Hub (the PDF requires links to them).

Dry run by default: prints exactly what would be uploaded. Nothing is sent unless
``--execute`` is given. Authentication comes from the usual HF token (``HF_TOKEN`` or
``huggingface-cli login``); no token is ever read from or written to this repository.

Examples (from the repository root on ADA):
    python scripts/upload_to_hf.py --namespace USER --part 2 --run outputs/part2/v2
    python scripts/upload_to_hf.py --namespace USER --part 1 --part1-root src/part1 --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PART2_OPTIMIZERS = ("adamw", "nadamw", "lion", "muon", "sophia")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def part1_plan(root: Path, namespace: str) -> list[dict]:
    plan = []
    for model_id in range(1, 6):
        folder = root / f"model {model_id}" / "full"
        files = {
            "checkpoint.pt": folder / "checkpoint.pt",
            "tokenizer.json": root / "shared" / "tokenizer_clean.json",
            "implementation.ipynb": root / "shared" / "implementation.ipynb",
        }
        for name in ("full_test_bleu.json", "full_test_predictions.jsonl"):
            if (folder / name).exists():
                files[name] = folder / name
        plan.append(
            {
                "repo": f"{namespace}/anlp-assignment2-part1-model{model_id}",
                "files": files,
                "card": (
                    f"# ANLP Assignment 2 - Part 1, model {model_id}\n\n"
                    "Decoder-only Transformer for Vietnamese/Japanese to English translation "
                    "(belumind/en-vi-ja-curated-500k-triplets). `checkpoint.pt` holds `config`, "
                    "`model` (state dict), `summary` and `tokenizer_sha256`; load it with the "
                    "`Config`/`Transformer` definitions in `implementation.ipynb`.\n"
                ),
            }
        )
    return plan


def part2_plan(run: Path, namespace: str) -> list[dict]:
    plan = []
    source = Path(__file__).resolve().parents[1] / "src" / "part2" / "model.py"
    for name in PART2_OPTIMIZERS:
        checkpoint = run / "checkpoints" / name / "fraction_1.0.pt"
        if not checkpoint.exists():
            continue
        plan.append(
            {
                "repo": f"{namespace}/anlp-assignment2-part2-{name}",
                "files": {
                    "checkpoint.pt": checkpoint,
                    "tokenizer.json": run / "tokenizer.json",
                    "resolved_config.yaml": run / "resolved_config.yaml",
                    "run_metadata.json": run / "run_metadata.json",
                    "metrics.csv": run / "metrics.csv",
                    "generations.csv": run / "generations.csv",
                    "model.py": source,
                },
                "card": (
                    f"# ANLP Assignment 2 - Part 2, dense LM trained with {name}\n\n"
                    "One pass over browndw/human-ai-parallel-corpus (train split by document "
                    "family). `checkpoint.pt` holds `model` (state dict), `model_configuration`, "
                    "`optimizer`, `grad_scaler` and token/step counters. Rebuild with "
                    "`DenseTransformer(ModelConfig(**checkpoint['model_configuration']))` from "
                    "`model.py`. `metrics.csv` lists every optimizer's 0.1-1.0 checkpoints.\n"
                ),
            }
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--namespace", required=True, help="HF user or organisation")
    parser.add_argument("--part", type=int, choices=(1, 2), required=True)
    parser.add_argument("--part1-root", default="src/part1")
    parser.add_argument("--run", default="outputs/part2/v2", help="Part 2 output directory")
    parser.add_argument("--visibility", choices=("private", "public"), default="private")
    parser.add_argument("--execute", action="store_true", help="actually upload")
    arguments = parser.parse_args()
    if "/" in arguments.namespace:
        raise SystemExit("namespace must be a bare user/organisation name")

    plan = (
        part1_plan(Path(arguments.part1_root), arguments.namespace)
        if arguments.part == 1
        else part2_plan(Path(arguments.run), arguments.namespace)
    )
    if not plan:
        raise SystemExit("nothing to upload: no finished checkpoints found")
    for entry in plan:
        missing = [str(path) for path in entry["files"].values() if not path.exists()]
        if missing:
            raise SystemExit(f"{entry['repo']}: missing {missing}")

    records = []
    api = None
    if arguments.execute:
        from huggingface_hub import HfApi

        api = HfApi()
        print(f"authenticated as {api.whoami()['name']}")
    for entry in plan:
        checkpoint_hash = sha256(entry["files"]["checkpoint.pt"])
        print(f"{'UPLOAD' if api else 'DRY RUN'} {entry['repo']} ({arguments.visibility})")
        for name, path in entry["files"].items():
            print(f"    {name:24s} <- {path} ({path.stat().st_size / 2**20:.1f} MiB)")
        if api is not None:
            api.create_repo(
                repo_id=entry["repo"],
                repo_type="model",
                private=arguments.visibility == "private",
                exist_ok=True,
            )
            api.upload_file(
                path_or_fileobj=(
                    entry["card"] + f"\ncheckpoint sha256: `{checkpoint_hash}`\n"
                ).encode(),
                path_in_repo="README.md",
                repo_id=entry["repo"],
            )
            for name, path in entry["files"].items():
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=name,
                    repo_id=entry["repo"],
                    commit_message=f"Upload {name}",
                )
        records.append(
            {
                "repo": entry["repo"],
                "url": f"https://huggingface.co/{entry['repo']}",
                "checkpoint_sha256": checkpoint_hash,
                "uploaded": api is not None,
            }
        )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
