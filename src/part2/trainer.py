"""Token-counted training loop shared by every optimizer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .data import DataBundle, fixed_training_order
from .evaluate import evaluate_continuation_bleu, evaluate_loss, precision_context


def learning_rate_scale(progress: float, warmup_fraction: float, min_ratio: float) -> float:
    if warmup_fraction > 0 and progress < warmup_fraction:
        return max(progress / warmup_fraction, 1e-8)
    decay_progress = (progress - warmup_fraction) / max(1.0 - warmup_fraction, 1e-12)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_learning_rate(optimizer, scale: float) -> None:
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
        group["lr"] = group["base_lr"] * scale


def current_learning_rates(optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def _cuda_peak(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    *,
    optimizer_name: str,
    fraction: float,
    tokens_seen: int,
    optimizer_steps: int,
    skipped_steps: int,
    initial_state_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "model_configuration": model.config.__dict__,
            "optimizer": optimizer.state_dict(),
            "grad_scaler": scaler.state_dict(),
            "optimizer_name": optimizer_name,
            "checkpoint_fraction": fraction,
            "training_tokens_seen": tokens_seen,
            "optimizer_steps": optimizer_steps,
            "skipped_amp_steps": skipped_steps,
            "initial_state_sha256": initial_state_sha256,
        },
        temporary,
    )
    temporary.replace(path)


def _sampled_label_hessian_update(model, optimizer, input_ids, device, precision, scaler) -> bool:
    """Sophia's Gauss-Newton-Bartlett estimate on the current batch.

    Labels are sampled from the model's own predictive distribution; the squared
    gradient of that loss is an unbiased estimate of the Gauss-Newton diagonal
    (Liu et al., 2023, Algorithm 2). Under FP16 the loss is multiplied by the current
    GradScaler scale and divided back inside ``update_hessian``; non-finite estimates
    are discarded. Returns whether the estimate was applied.
    """
    optimizer.zero_grad(set_to_none=True)
    with precision_context(device, precision):
        logits = model(input_ids)["logits"]
    flat = logits.reshape(-1, logits.shape[-1]).float()
    with torch.no_grad():
        sampled = torch.multinomial(torch.softmax(flat, dim=-1), 1).squeeze(-1)
    sampled_loss = torch.nn.functional.cross_entropy(flat, sampled)
    loss_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
    (sampled_loss * loss_scale).backward()
    applied = optimizer.update_hessian(gradient_divisor=loss_scale)
    optimizer.zero_grad(set_to_none=True)
    return applied


def train_optimizer(
    *,
    model,
    optimizer,
    optimizer_name: str,
    optimizer_hyperparameters: dict,
    data: DataBundle,
    config: dict,
    device: torch.device,
    output_directory: Path,
    initial_state_sha256: str,
) -> tuple[list[dict], list[dict]]:
    training = config["training"]
    evaluation = config["evaluation"]
    sequence_length = data.train.sequence_length
    requested_budget = training.get("token_budget")
    token_budget = data.train.training_tokens if requested_budget is None else int(requested_budget)
    if token_budget < sequence_length or token_budget > data.train.training_tokens:
        raise ValueError("token budget must be within one full training-dataset pass")
    if token_budget % sequence_length:
        raise ValueError("token budget must be divisible by sequence_length")
    example_budget = token_budget // sequence_length
    order = fixed_training_order(len(data.train), config["experiment"]["seed"])[:example_budget]
    fractions = training["evaluation_fractions"]
    target_examples = [math.ceil(example_budget * fraction) for fraction in fractions]
    if len(set(target_examples)) != len(target_examples):
        raise ValueError("dataset is too small to produce distinct evaluation checkpoints")
    checkpoint_fractions = training.get("checkpoint_fractions")
    if checkpoint_fractions is None:
        checkpoint_fractions = fractions
    saved_generations = evaluation.get("saved_generations_per_checkpoint")
    hessian_interval = optimizer_hyperparameters.get("hessian_interval")
    run_evaluations = evaluation.get("enabled", True)
    progress_path = output_directory / f"progress_{optimizer_name}.jsonl"
    progress_path.unlink(missing_ok=True)

    model.train()
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    precision = training["precision"]
    use_scaler = precision == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    metrics: list[dict] = []
    generations: list[dict] = []
    processed_examples = 0
    interval_loss_sum = 0.0
    interval_token_count = 0
    next_offset = 0
    batch_count = 0
    optimizer_steps = 0
    skipped_steps = 0
    interval_skipped = 0
    hessian_updates = 0
    hessian_rejected = 0
    evaluation_seconds = 0.0

    for fraction, target in zip(fractions, target_examples, strict=True):
        while processed_examples < target:
            size = min(training["batch_size"], target - processed_examples)
            indices = order[next_offset : next_offset + size]
            next_offset += size
            rows = data.train.blocks[indices].to(device)
            input_ids, labels = rows[:, :-1], rows[:, 1:]
            progress = ((processed_examples + size) * sequence_length) / token_budget
            lr_scale = learning_rate_scale(
                progress, training["warmup_fraction"], training["minimum_lr_ratio"]
            )
            set_learning_rate(optimizer, lr_scale)
            optimizer.zero_grad(set_to_none=True)
            with precision_context(device, precision):
                loss = model(input_ids, labels)["loss"]
            if use_scaler:
                scaler.scale(loss).backward()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # GradScaler lowers its scale exactly when it found inf/NaN gradients
                # and skipped optimizer.step(); the batch still counts as data seen.
                skipped = scaler.get_scale() < scale_before
            else:
                loss.backward()
                optimizer.step()
                skipped = False
            skipped_steps += int(skipped)
            interval_skipped += int(skipped)
            optimizer_steps += int(not skipped)
            batch_count += 1
            if hessian_interval and batch_count % hessian_interval == hessian_interval - 1:
                if _sampled_label_hessian_update(
                    model, optimizer, input_ids, device, precision, scaler
                ):
                    hessian_updates += 1
                else:
                    hessian_rejected += 1
            batch_tokens = int(labels.ne(-100).sum().item())
            processed_examples += size
            interval_loss_sum += float(loss.detach().float().item()) * batch_tokens
            interval_token_count += batch_tokens

        tokens_seen = processed_examples * sequence_length
        # Pure optimisation time: excludes all earlier checkpoint evaluations.
        training_seconds = time.perf_counter() - start_time - evaluation_seconds
        evaluation_start = time.perf_counter()
        row = {
            "optimizer": optimizer_name,
            "seed": config["experiment"]["seed"],
            "checkpoint_fraction": fraction,
            "evaluation_target_tokens": fraction * token_budget,
            "training_tokens_seen": tokens_seen,
            "training_loss": interval_loss_sum / interval_token_count,
            "optimizer_steps": optimizer_steps,
            "skipped_amp_steps": skipped_steps,
            "skipped_amp_steps_in_interval": interval_skipped,
            "grad_scale": float(scaler.get_scale()) if use_scaler else None,
            "hessian_updates": hessian_updates if hessian_interval else None,
            "hessian_updates_rejected": hessian_rejected if hessian_interval else None,
        }
        if run_evaluations:
            validation_loss, validation_perplexity = evaluate_loss(
                model,
                data.validation,
                batch_size=evaluation["batch_size"],
                device=device,
                precision=precision,
                max_examples=evaluation.get("max_validation_examples"),
            )
            bleu_start = time.perf_counter()
            test_bleu, checkpoint_generations = evaluate_continuation_bleu(
                model,
                data.test,
                data.tokenizer,
                optimizer_name=optimizer_name,
                training_tokens_seen=tokens_seen,
                checkpoint_fraction=fraction,
                device=device,
                prefix_fraction=evaluation["prefix_fraction"],
                generation_tokens=evaluation["generation_tokens"],
                max_examples=evaluation["bleu_examples"],
                batch_size=evaluation["generation_batch_size"],
            )
            # BLEU always uses every generated example; only the saved CSV is thinned
            # for intermediate checkpoints to keep the artefact small.
            if saved_generations is not None and fraction != fractions[-1]:
                checkpoint_generations = checkpoint_generations[:saved_generations]
            generations.extend(checkpoint_generations)
            row.update(
                {
                    "validation_loss": validation_loss,
                    "validation_perplexity": validation_perplexity,
                    "test_bleu": test_bleu,
                    "test_bleu_examples": len(data.test)
                    if evaluation["bleu_examples"] is None
                    else min(evaluation["bleu_examples"], len(data.test)),
                    "bleu_seconds": time.perf_counter() - bleu_start,
                }
            )
        elif fraction == fractions[-1]:
            # Sweep mode: only the end-of-budget validation loss is needed for selection.
            validation_loss, validation_perplexity = evaluate_loss(
                model,
                data.validation,
                batch_size=evaluation["batch_size"],
                device=device,
                precision=precision,
                max_examples=evaluation.get("max_validation_examples"),
            )
            row.update(
                {"validation_loss": validation_loss, "validation_perplexity": validation_perplexity}
            )
        evaluation_seconds += time.perf_counter() - evaluation_start
        if hasattr(optimizer, "clipped_fraction"):
            row["sophia_clipped_coordinate_fraction"] = optimizer.clipped_fraction()
        rates = current_learning_rates(optimizer)
        row.update(
            {
                "training_wall_clock_seconds": training_seconds,
                "wall_clock_seconds": time.perf_counter() - start_time,
                "peak_gpu_memory_if_available": _cuda_peak(device),
                "learning_rate": rates[0],
                "parameter_group_learning_rates": json.dumps(rates),
                "optimizer_hyperparameters": json.dumps(optimizer_hyperparameters, sort_keys=True),
            }
        )
        metrics.append(row)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "optimizer",
                        "checkpoint_fraction",
                        "training_loss",
                        "validation_loss",
                        "optimizer_steps",
                        "skipped_amp_steps",
                    )
                }
                | {"test_bleu": row.get("test_bleu")}
            ),
            flush=True,
        )
        interval_loss_sum = 0.0
        interval_token_count = 0
        interval_skipped = 0
        if training["save_checkpoints"] and any(
            math.isclose(fraction, value) for value in checkpoint_fractions
        ):
            _save_checkpoint(
                output_directory / "checkpoints" / optimizer_name / f"fraction_{fraction:.1f}.pt",
                model,
                optimizer,
                scaler,
                optimizer_name=optimizer_name,
                fraction=fraction,
                tokens_seen=tokens_seen,
                optimizer_steps=optimizer_steps,
                skipped_steps=skipped_steps,
                initial_state_sha256=initial_state_sha256,
            )
        model.train()
    if processed_examples * sequence_length != token_budget:
        raise AssertionError("training did not stop at the exact configured token budget")
    return metrics, generations
