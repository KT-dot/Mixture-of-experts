"""Explain the saved greedy vs beam-width-1 differences in outputs/part3.

Greedy (batch 128) and beam_1 (batch 32) produced different text for a small number of
stories. The decoders are the same algorithm (see src/part3/tests), so the hypothesis
is that different batch composition changes the left-padded width, FP32 reduction
order, and hence near-tie argmax decisions. Two checks, in FP32 (CPU by default; --device cuda reproduces the saved runs' GPU):

A. matched batching: for every 32-story batch containing a differing story, rerun
   greedy_decode and beam_search_decode(width 1) on exactly that batch and compare ids.
   On CUDA, the batch-32 greedy text is also compared with the saved beam_1 text.
B. tie margins: for each differing story, find the first token where the two saved
   continuations diverge and measure, with an unpadded batch-size-1 forward pass, the
   log-probability gap between the two competing tokens at that position.

Usage: python scripts/part3_greedy_beam1_diagnostic.py [--output outputs/part3]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.part3.config import load_config
from src.part3.decoding import IncrementalLM, beam_search_decode, greedy_decode
from src.part3.experiment import (
    build_masks,
    load_model_and_tokenizer,
    prepare_stories,
)


def read_generations(path: Path) -> dict[int, str]:
    with gzip.open(path, "rt") as handle:
        return {row["index"]: row["generation"] for row in map(json.loads, handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/part3.yaml")
    parser.add_argument("--output", default="outputs/part3")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, or cuda to match the saved runs")
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    output = Path(arguments.output)
    config = load_config(arguments.config)
    config["model"]["device"] = arguments.device
    metadata = json.loads((output / "run_metadata.json").read_text())

    greedy = read_generations(output / "generations" / "greedy.jsonl.gz")
    beam = read_generations(output / "generations" / "beam_1.jsonl.gz")
    differing = sorted(index for index in greedy if greedy[index] != beam[index])
    print(f"{len(differing)} of {len(greedy)} stories differ in the saved GPU outputs")

    device = torch.device(arguments.device)
    model, tokenizer = load_model_and_tokenizer(config, device)
    model.generate = None
    stories, excluded = prepare_stories(config, tokenizer, None)
    assert excluded == metadata["excluded_empty_rows"], "story set differs from the saved run"
    position = {story["index"]: offset for offset, story in enumerate(stories)}
    generation = config["generation"]
    logit_width = model.get_output_embeddings().weight.shape[0]
    stop_mask, banned_mask = build_masks(tokenizer, logit_width, generation["newline_policy"])
    stop_mask = stop_mask.to(device)
    banned_mask = None if banned_mask is None else banned_mask.to(device)
    lm = IncrementalLM(
        model,
        pad_token_id=tokenizer.eos_token_id,
        vocab_size=len(tokenizer),
        banned_mask=banned_mask,
    )

    # A. matched batching, using the beam run's batch boundaries (batch size 32).
    beam_batch = generation["beam_batch_size"]
    batch_starts = sorted({position[index] // beam_batch * beam_batch for index in differing})
    compared = identical = saved_matches = 0
    started = time.perf_counter()
    for count, start in enumerate(batch_starts, 1):
        prompts = [story["prompt_ids"] for story in stories[start : start + beam_batch]]
        kwargs = {"max_new_tokens": generation["max_new_tokens"], "stop_mask": stop_mask}
        a = greedy_decode(lm, prompts, **kwargs).sequences
        b = beam_search_decode(lm, prompts, beam_width=1, **kwargs).sequences
        compared += len(prompts)
        identical += sum(x == y for x, y in zip(a, b, strict=True))
        for story, ids in zip(stories[start : start + beam_batch], a, strict=True):
            content = ids[:-1] if ids and bool(stop_mask[ids[-1]]) else ids
            text = tokenizer.decode(content, skip_special_tokens=True).strip()
            saved_matches += text == beam[story["index"]]
        if count % 10 == 0:
            print(f"  A: {count}/{len(batch_starts)} batches, {time.perf_counter() - started:.0f}s")
    matched = {
        "batches": len(batch_starts),
        "stories_compared": compared,
        "identical_token_sequences": identical,
        # Meaningful only with --device cuda (same GPU type as the saved runs).
        "batch32_greedy_text_equal_to_saved_beam1": saved_matches,
    }
    print("A matched batching:", matched)

    # B. log-probability gap at the first divergence of the saved GPU continuations.
    gaps = []
    for index in differing:
        story = stories[position[index]]
        first = tokenizer(" " + greedy[index])["input_ids"]
        second = tokenizer(" " + beam[index])["input_ids"]
        split = next(
            (i for i, (x, y) in enumerate(zip(first, second, strict=False)) if x != y),
            min(len(first), len(second)),
        )
        if split >= min(len(first), len(second)):
            continue  # one text is a prefix of the other after re-tokenisation
        context = torch.tensor([story["prompt_ids"] + first[:split]], device=device)
        with torch.inference_mode():
            logits = model(input_ids=context).logits[0, -1, : len(tokenizer)].float()
        if banned_mask is not None:
            logits = logits.masked_fill(banned_mask[: len(tokenizer)], float("-inf"))
        log_probs = logits.log_softmax(-1)
        gaps.append(
            {
                "index": index,
                "divergence_token": split,
                "logprob_gap": abs(float(log_probs[first[split]] - log_probs[second[split]])),
                "both_tokens_top2": sorted(log_probs.topk(2).indices.tolist())
                == sorted([first[split], second[split]]),
            }
        )
    values = sorted(row["logprob_gap"] for row in gaps)
    margins = {
        "stories_measured": len(gaps),
        "median_logprob_gap": values[len(values) // 2] if values else None,
        "max_logprob_gap": values[-1] if values else None,
        "fraction_gap_below_1e-3": sum(v < 1e-3 for v in values) / len(values) if values else None,
        "fraction_competitors_are_top2": sum(r["both_tokens_top2"] for r in gaps) / len(gaps)
        if gaps
        else None,
    }
    print("B tie margins:", margins)
    result = {
        "differing_stories_in_saved_gpu_runs": len(differing),
        "stories_total": len(greedy),
        "device": arguments.device,
        "dtype": config["model"]["dtype"],
        "matched_batching": matched,
        "tie_margins": margins,
        "per_story": gaps,
    }
    (output / "greedy_beam1_diagnostic.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
