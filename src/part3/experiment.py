"""Orchestration for Part 3: decode ROCStories with Pythia-160M and score every run."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .decoding import (
    DecodeResult,
    IncrementalLM,
    beam_search_decode,
    greedy_decode,
    sample_decode,
)
from .metrics import (
    continuation_nll,
    corpus_perplexity,
    distinct_n,
    mean_and_sem,
    overlap_scores,
    positional_token_accuracy,
    repeated_trigram_fraction,
)
from .plotting import save_plots


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model_and_tokenizer(config: dict, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(settings["name"], revision=settings["revision"])
    dtype = {"float32": torch.float32, "float16": torch.float16}[settings["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(
        settings["name"], revision=settings["revision"], dtype=dtype
    )
    model.to(device).eval()
    return model, tokenizer


def newline_mask(tokenizer, logit_width: int) -> torch.Tensor:
    """Boolean mask over output ids of every token whose text contains a newline."""
    mask = torch.zeros(logit_width, dtype=torch.bool)
    for token_id in range(len(tokenizer)):
        if "\n" in tokenizer.decode([token_id]):
            mask[token_id] = True
    return mask


def build_masks(tokenizer, logit_width: int, newline_policy: str):
    """(stop_mask, banned_mask) for the configured newline policy.

    ``ban``: a story is one paragraph, so newline tokens are disallowed and only EOS
    or the length cap ends a continuation. (Pythia's most likely token after a
    sentence-final period is often a newline, so stopping on it would leave many
    greedy/beam continuations empty.) ``stop``: newline ends the continuation.
    """
    newlines = newline_mask(tokenizer, logit_width)
    stop = torch.zeros(logit_width, dtype=torch.bool)
    stop[tokenizer.eos_token_id] = True
    if newline_policy == "ban":
        return stop, newlines
    if newline_policy == "stop":
        return stop | newlines, None
    raise ValueError(f"unknown newline_policy {newline_policy!r}")


def prepare_stories(config: dict, tokenizer, limit: int | None) -> tuple[list[dict], list[int]]:
    """Tokenised (prompt, reference) pairs in split order, plus excluded row indices.

    A few upstream rows lack sentence punctuation, so the whole story landed in
    ``prompt`` and ``continuation`` is empty; they cannot be scored and are excluded.
    Original split indices are kept so every result traces back to its dataset row.
    """
    from datasets import load_dataset

    settings = config["dataset"]
    split = load_dataset(settings["name"], revision=settings["revision"], split=settings["split"])
    stories = []
    excluded = []
    for index, row in enumerate(split):
        if settings.get("max_samples") is not None and len(stories) >= settings["max_samples"]:
            break
        if limit is not None and len(stories) >= limit:
            break
        prompt = row[settings["prompt_field"]].strip()
        reference = row[settings["reference_field"]].strip()
        if not prompt or not reference:
            excluded.append(index)
            continue
        stories.append(
            {
                "index": index,
                "prompt": prompt,
                "reference": reference,
                "prompt_ids": tokenizer(prompt)["input_ids"],
                # A leading space makes the reference tokenise as it would follow the prompt.
                "reference_ids": tokenizer(" " + reference)["input_ids"],
            }
        )
    return stories, excluded


def _decode_batches(
    lm: IncrementalLM,
    stories: list[dict],
    strategy: dict,
    seed: int | None,
    *,
    max_new_tokens: int,
    stop_mask: torch.Tensor,
    batch_size: int,
) -> tuple[DecodeResult, float]:
    generator = None
    if strategy["type"] == "sample":
        generator = torch.Generator(device=lm.device).manual_seed(int(seed))
    sequences: list[list[int]] = []
    stopped: list[bool] = []
    forward_calls = 0
    if lm.device.type == "cuda":
        torch.cuda.synchronize(lm.device)
    start = time.perf_counter()
    for offset in range(0, len(stories), batch_size):
        prompts = [story["prompt_ids"] for story in stories[offset : offset + batch_size]]
        if strategy["type"] == "greedy":
            result = greedy_decode(lm, prompts, max_new_tokens=max_new_tokens, stop_mask=stop_mask)
        elif strategy["type"] == "beam":
            result = beam_search_decode(
                lm,
                prompts,
                beam_width=int(strategy["beam_width"]),
                max_new_tokens=max_new_tokens,
                stop_mask=stop_mask,
                length_penalty=float(strategy.get("length_penalty", 1.0)),
            )
        else:
            result = sample_decode(
                lm,
                prompts,
                max_new_tokens=max_new_tokens,
                stop_mask=stop_mask,
                generator=generator,
                temperature=float(strategy.get("temperature", 1.0)),
                top_k=strategy.get("top_k"),
                top_p=strategy.get("top_p"),
            )
        sequences.extend(result.sequences)
        stopped.extend(result.stopped)
        forward_calls += result.forward_calls
    if lm.device.type == "cuda":
        torch.cuda.synchronize(lm.device)
    elapsed = time.perf_counter() - start
    return DecodeResult(sequences, stopped, forward_calls), elapsed


def _score_run(
    model,
    tokenizer,
    stories: list[dict],
    result: DecodeResult,
    stop_ids: set[int],
    *,
    vocab_size: int,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    nll = continuation_nll(
        model,
        [story["prompt_ids"] for story in stories],
        result.sequences,
        pad_token_id=tokenizer.eos_token_id,
        vocab_size=vocab_size,
        batch_size=batch_size,
    )
    records = []
    precisions, recalls, f1s, accuracies = [], [], [], []
    texts = []
    for story, ids, stopped, (sum_nll, count) in zip(
        stories, result.sequences, result.stopped, nll, strict=True
    ):
        content_ids = ids[:-1] if ids and ids[-1] in stop_ids else ids
        text = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
        precision, recall, f1 = overlap_scores(text, story["reference"])
        accuracy = positional_token_accuracy(content_ids, story["reference_ids"])
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
        texts.append(text)
        records.append(
            {
                "index": story["index"],
                "generation": text,
                "generated_tokens": len(ids),
                "stopped": stopped,
                "nll": sum_nll,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )
    metrics = {}
    for name, values in (
        ("accuracy", accuracies),
        ("precision", precisions),
        ("recall", recalls),
        ("f1", f1s),
    ):
        mean, sem = mean_and_sem(values)
        metrics[name] = mean
        metrics[f"{name}_sem"] = sem
    metrics.update(
        {
            "generation_perplexity": corpus_perplexity(nll),
            "mean_generated_tokens": float(np.mean([len(ids) for ids in result.sequences])),
            "stopped_fraction": float(np.mean(result.stopped)),
            "empty_generation_fraction": float(np.mean([not text for text in texts])),
            "distinct_1": distinct_n(texts, 1),
            "distinct_2": distinct_n(texts, 2),
            "repeated_trigram_fraction": repeated_trigram_fraction(texts),
        }
    )
    return metrics, records


def _fingerprint(config: dict, stories: list[dict]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(config["model"], sort_keys=True).encode())
    digest.update(json.dumps(config["dataset"], sort_keys=True).encode())
    digest.update(json.dumps(config["generation"], sort_keys=True).encode())
    for story in stories:
        digest.update(json.dumps(story["prompt_ids"]).encode())
    return digest.hexdigest()[:16]


def _run_fingerprint(fingerprint: str, strategy: dict, seed: int | None) -> str:
    """Identity of one saved run: data/model/generation settings, this strategy's exact
    settings and seed, and the decoding/scoring source code. A changed strategy (e.g. a
    new ``top_k`` under the same name) or edited decoder therefore never reuses a result.
    """
    digest = hashlib.sha256(fingerprint.encode())
    digest.update(json.dumps(strategy, sort_keys=True).encode())
    digest.update(str(seed).encode())
    package = Path(__file__).resolve().parent
    for name in ("decoding.py", "metrics.py", "experiment.py"):
        digest.update((package / name).read_bytes())
    return digest.hexdigest()[:16]


def _reference_scores(model, tokenizer, stories, config, output: Path, fingerprint: str) -> dict:
    path = output / "reference_perplexity.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return cached
    nll = continuation_nll(
        model,
        [story["prompt_ids"] for story in stories],
        [story["reference_ids"] for story in stories],
        pad_token_id=tokenizer.eos_token_id,
        vocab_size=len(tokenizer),
        batch_size=config["generation"]["scoring_batch_size"],
    )
    lengths = [len(story["reference_ids"]) for story in stories]
    scores = {
        "fingerprint": fingerprint,
        "stories": len(stories),
        "reference_perplexity": corpus_perplexity(nll),
        "reference_tokens_total": int(sum(lengths)),
        "reference_tokens_mean": float(np.mean(lengths)),
        "reference_tokens_p95": float(np.percentile(lengths, 95)),
        "reference_tokens_p99": float(np.percentile(lengths, 99)),
        "reference_tokens_max": int(max(lengths)),
        "fraction_references_longer_than_max_new_tokens": float(
            np.mean([length > config["generation"]["max_new_tokens"] for length in lengths])
        ),
    }
    path.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    return scores


def _runs_for(strategy: dict) -> list[tuple[str, int | None]]:
    if strategy["type"] == "sample":
        return [(f"{strategy['name']}_seed{seed}", int(seed)) for seed in strategy["seeds"]]
    return [(strategy["name"], None)]


def run_beam_timing(
    lm: IncrementalLM, stories: list[dict], config: dict, stop_mask: torch.Tensor, output: Path
) -> list[dict]:
    """Beam-width timing on one fixed subset of test stories.

    For each batch size (number of prompts decoded together) and each width, the same
    stories are decoded with the main-run settings. Every row records wall-clock time,
    forward calls and peak memory, so per-step cost can be separated from step count.
    """
    timing = config["timing"]
    subset = stories[: timing["stories"]]
    max_new_tokens = config["generation"]["max_new_tokens"]
    rows = []
    for batch_size in timing["batch_sizes"]:
        warmup = {"type": "beam", "beam_width": max(timing["widths"])}
        _decode_batches(
            lm,
            subset[:batch_size],
            warmup,
            None,
            max_new_tokens=max_new_tokens,
            stop_mask=stop_mask,
            batch_size=batch_size,
        )
        for width in timing["widths"]:
            strategy = {"type": "beam", "beam_width": width, "length_penalty": 1.0}
            if lm.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(lm.device)
            result, seconds = _decode_batches(
                lm,
                subset,
                strategy,
                None,
                max_new_tokens=max_new_tokens,
                stop_mask=stop_mask,
                batch_size=batch_size,
            )
            generated = sum(len(ids) for ids in result.sequences)
            rows.append(
                {
                    "batch_size": batch_size,
                    "beam_width": width,
                    "rows_per_forward": batch_size * width,
                    "stories": len(subset),
                    "seconds": seconds,
                    "ms_per_story": 1000 * seconds / len(subset),
                    "forward_calls": result.forward_calls,
                    "ms_per_forward_call": 1000 * seconds / result.forward_calls,
                    "mean_output_tokens": generated / len(subset),
                    "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(lm.device))
                    if lm.device.type == "cuda"
                    else None,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
            pd.DataFrame(rows).to_csv(output / "beam_timing.csv", index=False)
    return rows


def greedy_beam1_agreement(output: Path) -> dict | None:
    """Fraction of stories where beam_1 and greedy produced identical text.

    They are the same algorithm (see tests); on GPU, different batch sizes change the
    padded width and hence FP32 reduction order, which can flip a near-tie argmax.
    """
    paths = [output / "generations" / f"{name}.jsonl.gz" for name in ("greedy", "beam_1")]
    if not all(path.exists() for path in paths):
        return None
    texts = []
    for path in paths:
        with gzip.open(path, "rt") as handle:
            texts.append([json.loads(line)["generation"] for line in handle])
    if len(texts[0]) != len(texts[1]):
        return None
    same = sum(a == b for a, b in zip(*texts, strict=True))
    result = {"stories": len(texts[0]), "identical": same, "fraction": same / len(texts[0])}
    (output / "greedy_beam1_agreement.json").write_text(json.dumps(result, indent=2))
    return result


def aggregate(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))["metrics"]
        for path in sorted((output / "runs").glob("*.json"))
    ]
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs, runs
    runs = runs.sort_values(["order", "seed"], na_position="first")
    runs.to_csv(output / "metrics_by_run.csv", index=False)
    value_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "generation_perplexity",
        "mean_generated_tokens",
        "stopped_fraction",
        "empty_generation_fraction",
        "distinct_1",
        "distinct_2",
        "repeated_trigram_fraction",
        "decode_seconds",
    ]
    grouped = runs.groupby(["order", "strategy", "type"], sort=True)
    summary = grouped[value_columns].mean()
    spread = grouped[value_columns].std(ddof=1).add_suffix("_seed_std")
    summary = summary.join(spread).join(grouped.size().rename("runs")).reset_index()
    first = grouped.first()
    for column in (
        "beam_width",
        "top_k",
        "top_p",
        "temperature",
        "reference_perplexity",
        "stories",
    ):
        summary[column] = first[column].to_numpy() if column in first else None
    summary.to_csv(output / "summary.csv", index=False)
    return runs, summary


def run_experiment(
    config: dict,
    *,
    strategy_names: list[str] | None = None,
    limit: int | None = None,
    output_directory: str | Path | None = None,
) -> dict:
    output = Path(output_directory or config["output"]["directory"]).expanduser().resolve()
    (output / "runs").mkdir(parents=True, exist_ok=True)
    (output / "generations").mkdir(parents=True, exist_ok=True)
    seed_everything(config["experiment"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = resolve_device(config["model"].get("device", "auto"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable on this node")
    model, tokenizer = load_model_and_tokenizer(config, device)
    # generate() is deliberately unavailable: every strategy must go through forward().
    model.generate = None
    stories, excluded_rows = prepare_stories(config, tokenizer, limit)
    minimum = config["dataset"].get("min_samples", 1000)
    if limit is None and len(stories) < minimum:
        raise ValueError(f"only {len(stories)} stories; the assignment requires {minimum}")
    generation = config["generation"]
    logit_width = model.get_output_embeddings().weight.shape[0]
    stop_mask, banned_mask = build_masks(tokenizer, logit_width, generation["newline_policy"])
    stop_mask = stop_mask.to(device)
    banned_mask = None if banned_mask is None else banned_mask.to(device)
    stop_ids = set(torch.nonzero(stop_mask).flatten().cpu().tolist())
    lm = IncrementalLM(
        model,
        pad_token_id=tokenizer.eos_token_id,
        vocab_size=len(tokenizer),
        banned_mask=banned_mask,
    )
    fingerprint = _fingerprint(config, stories)
    reference = _reference_scores(model, tokenizer, stories, config, output, fingerprint)
    metadata = {
        "fingerprint": fingerprint,
        "model": config["model"],
        "dataset": config["dataset"],
        "stories_evaluated": len(stories),
        "excluded_empty_rows": excluded_rows,
        "limit": limit,
        "generation": generation,
        "stop_token_ids": sorted(stop_ids),
        "banned_token_count": 0 if banned_mask is None else int(banned_mask.sum()),
        "tokenizer_vocab_size": len(tokenizer),
        "model_logit_width": logit_width,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "reference": reference,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    selected = set(strategy_names) if strategy_names else None
    for order, strategy in enumerate(config["strategies"]):
        if selected is not None and strategy["name"] not in selected:
            continue
        for run_name, seed in _runs_for(strategy):
            path = output / "runs" / f"{run_name}.json"
            run_fingerprint = _run_fingerprint(fingerprint, strategy, seed)
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous["metrics"].get("run_fingerprint") == run_fingerprint:
                    print(f"skip {run_name}: already complete", flush=True)
                    continue
            batch_size = (
                generation["beam_batch_size"]
                if strategy["type"] == "beam"
                else generation["batch_size"]
            )
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            result, seconds = _decode_batches(
                lm,
                stories,
                strategy,
                seed,
                max_new_tokens=generation["max_new_tokens"],
                stop_mask=stop_mask,
                batch_size=batch_size,
            )
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            metrics, records = _score_run(
                model,
                tokenizer,
                stories,
                result,
                stop_ids,
                vocab_size=len(tokenizer),
                batch_size=generation["scoring_batch_size"],
            )
            metrics.update(
                {
                    "run": run_name,
                    "strategy": strategy["name"],
                    "type": strategy["type"],
                    "order": order,
                    "seed": seed,
                    "beam_width": strategy.get("beam_width"),
                    "top_k": strategy.get("top_k"),
                    "top_p": strategy.get("top_p"),
                    "temperature": strategy.get("temperature", 1.0)
                    if strategy["type"] == "sample"
                    else None,
                    "length_penalty": strategy.get("length_penalty")
                    if strategy["type"] == "beam"
                    else None,
                    "stories": len(stories),
                    "batch_size": batch_size,
                    "decode_seconds": seconds,
                    "ms_per_story": 1000 * seconds / len(stories),
                    "forward_calls": result.forward_calls,
                    "peak_gpu_memory_bytes": peak,
                    "reference_perplexity": reference["reference_perplexity"],
                    "fingerprint": fingerprint,
                    "run_fingerprint": run_fingerprint,
                }
            )
            with gzip.open(output / "generations" / f"{run_name}.jsonl.gz", "wt") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            path.write_text(json.dumps({"metrics": metrics}, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        key: metrics[key]
                        for key in (
                            "run",
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "generation_perplexity",
                            "decode_seconds",
                        )
                    }
                ),
                flush=True,
            )

    if config.get("timing") and (selected is None or "beam_timing" in selected):
        run_beam_timing(lm, stories, config, stop_mask, output)
    greedy_beam1_agreement(output)
    runs, summary = aggregate(output)
    save_plots(output)
    references_path = output / "references.jsonl.gz"
    with gzip.open(references_path, "wt") as handle:
        for story in stories:
            handle.write(
                json.dumps({key: story[key] for key in ("index", "prompt", "reference")}) + "\n"
            )
    return {
        "metrics": runs.to_dict("records"),
        "summary": summary.to_dict("records"),
        "output_directory": str(output),
    }
