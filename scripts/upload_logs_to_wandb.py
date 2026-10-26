"""Publish the saved training logs of Parts 1 and 2 to Weights & Biases.

Training ran before W&B logging was required, so every run here is rebuilt from the records
the original SLURM jobs saved: Part 1 from the result record embedded in each executed model
notebook plus the per-step history stored in each checkpoint (``model N/full/history.json``,
exported from ``checkpoint.pt['history']``), Part 2 from ``progress_*.jsonl``. Each run is tagged ``uploaded-from-saved-logs``,
names its SLURM job, and carries that job's log file. Nothing is recomputed.

Dry run by default (prints the plan); ``--execute`` uploads. Run on ADA from the repository root:
    python scripts/upload_logs_to_wandb.py [--entity ENTITY] [--project anlp-assignment2] --execute
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HOME = Path.home() / "anlp_a2"
PART1 = HOME / "part1_20260922"
PART1_JOB = "2711535"
OPTIMIZERS = ("adamw", "nadamw", "lion", "muon", "sophia")
TAG = "uploaded-from-saved-logs"


def part1_result(model: int) -> dict:
    notebook = PART1 / f"model {model}" / f"model_{model}.executed.ipynb"
    for cell in json.loads(notebook.read_text())["cells"]:
        if "part1-result" not in cell.get("metadata", {}).get("tags", []):
            continue
        for output in cell.get("outputs", []):
            record = output.get("data", {}).get("application/json")
            if record and record.get("status") == "full run completed":
                return record
    raise SystemExit(f"no full-run result in {notebook}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_hashes(run: Path) -> dict[str, str]:
    path = run / "checkpoint_sha256.txt"
    if not path.exists():
        return {}
    return {line.split()[1].split("/")[-2]: line.split()[0] for line in path.read_text().splitlines()}


def part1_plan() -> list[dict]:
    full_bleu = {row["model"]: row for row in json.loads((PART1 / "shared" / "full_test_bleu.json").read_text())}
    plan = []
    for model in range(1, 6):
        result = part1_result(model)
        # The per-step history lives in the checkpoint (checkpoint.pt['history']), exported to JSON
        # beside it; the run record only keeps summaries.
        saved = json.loads((PART1 / f"model {model}" / "full" / "history.json").read_text())
        if saved["checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise SystemExit(f"model {model}: history.json is not from the recorded checkpoint")
        points = []
        for row in sorted(saved["history"], key=lambda r: (r["input_tokens"], r["kind"])):
            prefix = "train" if row["kind"] == "train" else "validation"
            point = {"input_tokens": row["input_tokens"]}
            point.update({f"{prefix}/{k}": v for k, v in row.items() if k not in ("kind", "input_tokens")})
            points.append(point)
        bleu = full_bleu[model]
        summary = {f"test/{k}": v for k, v in result["test"].items()}
        summary |= {f"bleu_1000_sample/{k}": result["bleu"][k] for k in ("bleu", "bleu_vi", "bleu_ja")}
        summary |= {f"bleu_full_test/{k}": bleu[k] for k in ("bleu", "bleu_vi", "bleu_ja", "triplets")}
        summary["bleu_full_test/ci95_low"], summary["bleu_full_test/ci95_high"] = bleu["bleu_ci95"]
        plan.append(
            {
                "name": f"part1-model{model}",
                "group": "part1-moe",
                "tags": ["part1", TAG],
                "x": "input_tokens",
                "notes": (
                    f"Part 1 model {model} ({result['name']}). Full training run, SLURM job "
                    f"{PART1_JOB}_{model}, uploaded after training: curves from the history saved in "
                    f"the run's checkpoint.pt, summaries from model_{model}.executed.ipynb. "
                    "x-axis: non-padding training input tokens."
                ),
                "config": {
                    **result["config"],
                    "ffn": result["name"],
                    "parameters": result["parameters"],
                    "training_input_tokens": result["training"]["input_tokens"],
                    "training_target_tokens": result["training"]["target_tokens"],
                    "training_steps": result["training"]["steps"],
                    "training_seconds": result["training"]["seconds"],
                    "schedule_digest": result["training"]["schedule_digest"],
                    "dataset_revision": result["dataset_revision"],
                    "tokenizer_sha256": result["tokenizer_sha256"],
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "huggingface": f"https://huggingface.co/Devatri/anlp-assignment2-part1-model{model}",
                    "slurm_job": f"{PART1_JOB}_{model}",
                },
                "points": points,
                "summary": summary,
                "tables": {"lr_pilots": result["pilots"]},
                "files": [PART1 / f"model-{PART1_JOB}-{model}.log",
                          PART1 / f"model {model}" / f"model_{model}.executed.ipynb"],
            }
        )
    return plan


def part2_entry(run: Path, optimizer: str, name: str, group: str, job: str, notes: str, hashes: dict) -> dict:
    rows = sorted(read_jsonl(run / f"progress_{optimizer}.jsonl"), key=lambda r: r["checkpoint_fraction"])
    points = []
    for row in rows:
        point = {
            "training_tokens": row["training_tokens_seen"],
            "checkpoint_fraction": row["checkpoint_fraction"],
            "train/loss_interval_mean": row["training_loss"],
            "validation/loss": row["validation_loss"],
            "validation/perplexity": row["validation_perplexity"],
            "test/bleu": row["test_bleu"],
            "optimizer/learning_rate": row["learning_rate"],
            "optimizer/steps": row["optimizer_steps"],
            "amp/grad_scale": row["grad_scale"],
            "amp/skipped_steps": row["skipped_amp_steps"],
            "time/training_wall_clock_seconds": row["training_wall_clock_seconds"],
        }
        if row.get("hessian_updates") is not None:
            point["optimizer/hessian_updates"] = row["hessian_updates"]
            point["optimizer/hessian_updates_rejected"] = row["hessian_updates_rejected"]
        points.append(point)
    metadata = json.loads((run / "run_metadata.json").read_text())
    config = {
        "optimizer": optimizer,
        "optimizer_hyperparameters": json.loads(rows[-1]["optimizer_hyperparameters"]),
        "seed": rows[-1]["seed"],
        "model_configuration": metadata.get("model_configuration"),
        "data_order_seed": metadata.get("data_order_seed"),
        "dataset": metadata.get("data", {}).get("source"),
        "dataset_revision": metadata.get("data", {}).get("resolved_revision"),
        "initial_state_sha256": metadata.get("initial_state_sha256"),
        "slurm_job": job,
    }
    if optimizer in hashes:
        config["checkpoint_sha256"] = hashes[optimizer]
    if group == "part2-optimizers":
        config["huggingface"] = f"https://huggingface.co/Devatri/anlp-assignment2-part2-{optimizer}"
    final = rows[-1]
    return {
        "name": name,
        "group": group,
        "tags": ["part2", optimizer, TAG],
        "x": "training_tokens",
        "notes": notes,
        "config": config,
        "points": points,
        "summary": {"final/validation_loss": final["validation_loss"], "final/test_bleu": final["test_bleu"],
                    "final/validation_perplexity": final["validation_perplexity"]},
        "tables": {},
        "files": sorted(Path("logs").glob(f"part2*-{job}.log"))
        + [p for p in (run / "resolved_config.yaml", run / "run_metadata.json") if p.exists()],
    }


def part2_plan(run: Path) -> list[dict]:
    hashes = checkpoint_hashes(run)
    plan = [
        part2_entry(
            run, name, f"part2-{name}", "part2-optimizers", "2721621",
            f"Part 2, dense LM trained with {name} for one pass (1.0x dataset), evaluated every 0.1x. "
            "SLURM job 2721621, uploaded after training from progress_" + name + ".jsonl. "
            "train/loss_interval_mean is the token-weighted mean training loss since the previous checkpoint.",
            hashes,
        )
        for name in OPTIMIZERS
    ]
    for lr, job in (("0.0001", "2721708"), ("0.0002", "2721709")):
        check = run.parent / f"{run.name}_lion_check_lr{lr}"
        plan.append(
            part2_entry(
                check, "lion", f"part2-lion-lr{lr}-check", "part2-lion-lr-check", job,
                f"Extra full-pass Lion run at lr {lr} (same initial weights and tokenizer), a sensitivity "
                f"check next to the sweep-selected run; not part of the selection protocol. SLURM job {job}, "
                "uploaded after training from progress_lion.jsonl.",
                {},
            )
        )
    sweep = run / "sweep"
    with (sweep / "sweep.csv").open() as handle:
        trials = list(csv.DictReader(handle))
    selected = json.loads((sweep / "selected_hyperparameters.json").read_text())
    plan.append(
        {
            "name": "part2-lr-sweep",
            "group": "part2-lr-sweep",
            "tags": ["part2", "sweep", TAG],
            "x": None,
            "notes": (
                f"Learning-rate sweep: {len(trials)} short-budget proxy trials (one evaluation each), "
                "SLURM job 2721621, uploaded after training from sweep/sweep.csv."
            ),
            "config": {"selected": selected},
            "points": [],
            "summary": {f"selected_lr/{k}": v["selected_lr"] for k, v in selected.items()},
            "tables": {"sweep_trials": trials},
            "images": {"sweep_validation_loss_vs_lr": sweep / "sweep_validation_loss_vs_lr.png"},
            "files": [sweep / "sweep.csv", sweep / "selected_hyperparameters.json"],
        }
    )
    return plan


def upload(entry: dict, entity: str | None, project: str) -> str:
    import wandb

    run = wandb.init(entity=entity, project=project, name=entry["name"], group=entry["group"],
                     tags=entry["tags"], notes=entry["notes"], config=entry["config"], job_type="train",
                     reinit="finish_previous")
    if entry["x"]:
        run.define_metric(entry["x"])
        for prefix in {key.split("/")[0] for point in entry["points"] for key in point if "/" in key}:
            run.define_metric(f"{prefix}/*", step_metric=entry["x"])
    for point in entry["points"]:
        run.log(point)
    for key, rows in entry["tables"].items():
        columns = list(rows[0])
        run.log({key: wandb.Table(columns=columns, data=[[row[c] for c in columns] for row in rows])})
    for key, path in entry.get("images", {}).items():
        run.log({key: wandb.Image(str(path))})
    for key, value in entry["summary"].items():
        run.summary[key] = value
    for path in entry["files"]:
        run.save(str(Path(path).resolve()), base_path=str(Path(path).resolve().parent), policy="now")
    url = run.url
    run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entity", default=None, help="W&B entity (default: the API key's default)")
    parser.add_argument("--project", default="anlp-assignment2")
    parser.add_argument("--part2-run", default="outputs/part2/v2")
    parser.add_argument("--only", nargs="+", help="upload only these run names")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()

    plan = part1_plan() + part2_plan(Path(arguments.part2_run))
    if arguments.only:
        plan = [entry for entry in plan if entry["name"] in arguments.only]
    for entry in plan:
        missing = [str(p) for p in entry["files"] + list(entry.get("images", {}).values()) if not Path(p).exists()]
        if missing:
            raise SystemExit(f"{entry['name']}: missing {missing}")
    records = []
    for entry in plan:
        print(f"{'UPLOAD' if arguments.execute else 'DRY RUN'} {entry['group']}/{entry['name']}: "
              f"{len(entry['points'])} points, tables {list(entry['tables'])}, "
              f"files {[Path(p).name for p in entry['files']]}")
        url = upload(entry, arguments.entity, arguments.project) if arguments.execute else None
        records.append({"run": entry["name"], "group": entry["group"], "url": url})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
