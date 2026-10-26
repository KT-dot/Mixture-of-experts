# Advanced NLP — Assignment 2

Code for the three tasks of Assignment 2:

| Part | Task | Where | How it runs |
|---|---|---|---|
| 1 | Mixture-of-Experts FFN variants for vi/ja → en translation | `src/part1/` (notebooks) | papermill on ADA |
| 2 | Hand-written optimizers (AdamW, NAdamW, Lion, Muon; bonus Sophia) | `src/part2/`, `configs/part2.yaml` | `python main.py --part 2` |
| 3 | Decoding strategies on pretrained Pythia-160M | `src/part3/`, `configs/part3.yaml` | `python main.py --part 3` |

The report (`report.pdf`) is written separately by the student. Model checkpoints are
not in git (see `.gitignore`); they are published on Hugging Face:

- Part 1 (one repository per FFN variant; checkpoint, tokenizer, implementation, full-test BLEU):
  - Model 1: https://huggingface.co/Devatri/anlp-assignment2-part1-model1
  - Model 2: https://huggingface.co/Devatri/anlp-assignment2-part1-model2
  - Model 3: https://huggingface.co/Devatri/anlp-assignment2-part1-model3
  - Model 4: https://huggingface.co/Devatri/anlp-assignment2-part1-model4
  - Model 5: https://huggingface.co/Devatri/anlp-assignment2-part1-model5
- Part 2 (one repository per optimizer; final checkpoint, tokenizer, config, metrics, generations, `model.py`):
  - AdamW: https://huggingface.co/Devatri/anlp-assignment2-part2-adamw
  - NAdamW: https://huggingface.co/Devatri/anlp-assignment2-part2-nadamw
  - Lion: https://huggingface.co/Devatri/anlp-assignment2-part2-lion
  - Muon: https://huggingface.co/Devatri/anlp-assignment2-part2-muon
  - Sophia (bonus): https://huggingface.co/Devatri/anlp-assignment2-part2-sophia

Publication runs from ADA with `bash scripts/ada_finalize.sh <hf-namespace> public` (or, for Part 2
alone, `sbatch scripts/hf_upload_part2_ada.sbatch <hf-namespace> public`). It records Part 2
checkpoint hashes, submits the greedy/beam-1 diagnostic, and runs `scripts/upload_to_hf.py`, which
is a dry run unless `--execute` is passed. The SHA-256 of every published Part 2 checkpoint is in
`assets/results/part2/v2/checkpoint_sha256.txt`; `hf_part2.json` is the upload record.

## Environments

Two uv projects, both Python 3.12:

- **Root project** (Parts 2 and 3): `uv sync` in the repository root, using `pyproject.toml` and `uv.lock`.
  Run the tests with `uv run pytest`, which covers `src/part2/tests` and `src/part3/tests`.
- **Part 1**: a separate project in `src/part1/` (`src/part1/pyproject.toml`, `src/part1/uv.lock`);
  run `uv sync` inside `src/part1/`. It pins the notebook toolchain (`papermill`, `ipykernel`) and
  the CUDA 12.4 PyTorch wheel.

The runs reported here used the ADA cluster (RTX 2080 Ti, 11 GB, FP16 autocast with
`GradScaler` for training, FP32 for Part 3 inference). The SLURM scripts are in `scripts/`.

## Part 1 — MoE translation (`src/part1/`)

`shared/implementation.ipynb` holds the model, data pipeline, training loop, evaluation and
routing diagnostics. The other notebooks load its cells tagged `definitions`. Run the notebooks
with the working directory inside `src/part1/` (Jupyter's default when a notebook is opened
from its folder); they locate `shared/implementation.ipynb` from there.

1. `00_setup_and_checks.ipynb`: downloads the revision-pinned dataset, trains the shared byte-level BPE tokenizer on train only, and runs correctness checks.
2. `model N/model_N.ipynb` for N = 1…5, run with `PART1_PROFILE=full`: pilot LR selection on validation NLL, the full run, then test perplexity, BLEU and routing heatmaps.
   Variants:
   1. Dense MLP.
   2. 4 experts, top-1.
   3. 4 experts, top-2.
   4. 1 shared expert + 3 routed experts, top-1.
   5. 4 experts, top-2, matching the dense model's active parameters.
3. `06_comparison.ipynb`: cross-model tables and plots.
4. `10_full_test_bleu.ipynb`: re-scores every saved checkpoint on the **entire** test split (the model notebooks use a seeded 1,000-triplet sample). It also reports BLEU restricted to test triplets whose sources never appear verbatim in train.
5. `07_execution_and_upload.ipynb`: publishes checkpoints. `08_translate_locally.ipynb`: interactive translation. `09_reproducibility_checks.ipynb`: resume/determinism checks.

Variants 1–4 have exactly 27,079,680 parameters. Variant 5 has 27,079,680 active parameters
(39,650,304 total). All five use the same 12,473-batch schedule (29,239,313 supervised target
tokens).

## Part 2 — optimizers (`src/part2/`)

Every optimizer subclasses only `torch.optim.Optimizer`; no `torch.optim` update rule is
called.

| Category (Wen et al., 2025, Table 1) | Optimizer | File |
|---|---|---|
| Baseline | AdamW | `optimizers/adamw.py` |
| Variance-reduced AdamW variant | NAdamW | `optimizers/nadamw.py` |
| Memory-efficient | Lion | `optimizers/lion.py` |
| Matrix-based | Muon (+ hand-written AdamW for non-matrix tensors) | `optimizers/muon.py` |
| Hessian-approximation (bonus) | SophiaG (Gauss-Newton-Bartlett estimator) | `optimizers/sophia.py` |

The model is Part 1's dense variant (d=512, 6 layers, 8 heads, 2048 hidden), trained from
one shared initial state on `browndw/human-ai-parallel-corpus`. The corpus has one official
split; it is divided 90/5/5 by source-document family so parallel human/AI versions never
cross splits. Each optimizer gets exactly one pass over the training blocks (1× the dataset).
Validation loss and test continuation BLEU are recorded at every 0.1× of that budget. BLEU
greedily completes the second half of every 256-token test block from its first half, and is
scored on **all** test blocks at every checkpoint.

```bash
python main.py --part 2 --synthetic-smoke                     # CPU-only pipeline check
python main.py --part 2 --sweep --output outputs/part2/v2     # bonus LR sweep (0.1x-budget proxies, validation loss; grid auto-extends x2 past an edge minimum, resumable)
python main.py --part 2 --hyperparameters outputs/part2/v2/sweep/selected_hyperparameters.json \
               --output outputs/part2/v2                      # full runs at the selected LRs
sbatch scripts/part2_ada.sbatch                               # both steps on ADA
```

Outputs:

- `metrics.csv` / `metrics.json`: one row per optimizer × checkpoint, including optimizer steps, skipped AMP steps and pure training time.
- Plots: `validation_loss_vs_tokens.png`, `test_bleu_vs_tokens.png`.
- `generations.csv`, `progress_<optimizer>.jsonl`.
- `sweep/sweep.csv`, `sweep/sweep_validation_loss_vs_lr.png`, `sweep/selected_hyperparameters.json`.
- `checkpoints/<optimizer>/fraction_1.0.pt`.

## Part 3 — decoding strategies (`src/part3/`)

`EleutherAI/pythia-160m` (revision-pinned, FP32) on `hamishivi/ROCStories`. The whole
**test split** is used: 19,633 rows, minus 22 whose `continuation` is empty upstream. Those rows are
excluded and their indices recorded in `run_metadata.json`. `prompt` is each story's first
sentence and the model continues it. `model.generate()` is never used: `decoding.py` drives
`model.forward()` step by step with a KV cache, left padding and explicit position ids.

- Strategies:
  - Greedy.
  - Beam search, widths 1, 2 and 4, length penalty 1.
  - Top-k with k ∈ {5, 20, 50}.
  - Top-p with p ∈ {0.5, 0.9, 0.95}.
  - Untruncated sampling.

  Each sampling strategy runs with 3 seeds.
- Output constraints: newline tokens are banned for every strategy, because a story is one paragraph. A continuation ends at EOS or after 64 tokens.
- Beam-width timing: every width is timed on the full batched run. A fixed 256-story subset is also timed at 1 and 32 prompts per batch, for widths 1, 2, 4, 8 and 16.

Metric definitions (also in `src/part3/metrics.py`):

- **Perplexity**: `exp(total NLL / tokens)` under the unmodified model (T=1, no truncation). It is computed for the reference continuation (independent of strategy) and for each strategy's own generations.
- **Accuracy**: position-wise token accuracy against the reference continuation's Pythia tokens.
- **Precision / recall / F1**: bag-of-words overlap (lower-cased, punctuation removed) between generation and reference, per story, then macro-averaged.

Diversity statistics (distinct-1/2, repeated-trigram fraction) and length and stopping
statistics are recorded alongside.

```bash
python main.py --part 3                                   # everything (resumes finished runs)
python main.py --part 3 --strategies greedy beam_4        # a subset
python main.py --part 3 --limit 64 --output outputs/p3dbg # quick debug run
sbatch scripts/part3_ada.sbatch
```

`scripts/part3_greedy_beam1_diagnostic.py` (`sbatch scripts/part3_diagnostic_ada.sbatch` on ADA) explains
the few stories where the saved greedy (batch 128) and beam-1 (batch 32) texts differ. It reruns
both decoders on identical batches and measures the log-probability gap at each first divergence.

Outputs:

- `summary.csv` (mean and seed std per strategy), `metrics_by_run.csv`.
- `runs/*.json`, `generations/*.jsonl.gz`.
- `reference_perplexity.json`, `beam_timing.csv`.
- Plots: `decoding_metrics.png`, `quality_vs_diversity.png`, `beam_width_timing.png`.

## Results in this repository

`outputs/` is not committed. The small result files are copied into `assets/results/`
(tables, metadata and figures only; checkpoints and generations are on Hugging Face / ADA):

| Path | Contents |
|---|---|
| `assets/results/part1/full_test_bleu.json` | Full-test-split BLEU (vi, ja, combined, bootstrap CI, unseen-source subset) for all five Part 1 checkpoints, with checkpoint SHA-256s. Other Part 1 measurements live in the executed `src/part1/model N/model_N.ipynb` and `src/part1/06_comparison.ipynb`. |
| `assets/results/part2/v2/` | Final Part 2 run: `metrics.csv` (5 optimizers × 10 checkpoints), both token-axis plots, run and dataset metadata, Muon parameter routing, published checkpoint SHA-256s and the Hugging Face upload record. |
| `assets/results/part2/v2/sweep/` | Learning-rate sweep trials, the selected learning rates, and the sweep plot. |
| `assets/results/part2/lion_lr_check/` | Extra full-pass Lion runs at lr 1e-4 and 2e-4 (same initial weights and tokenizer). These are a sensitivity check next to the sweep-selected 4e-4 run and are not part of the selection protocol. |
| `assets/results/part3/` | `summary.csv` (per strategy, seed mean/std), `metrics_by_run.csv`, `beam_timing.csv`, reference perplexity, run metadata, the greedy vs beam-1 agreement and diagnostic, and the three figures. |

## Tests

`uv run pytest` runs the test suite:

- **Optimizers**: AdamW, NAdamW, Lion and Muon (matrix path) are checked against independent scalar re-implementations of their equations over several steps, including gradient clipping. Sophia's clipped ratio is checked by hand computation. Newton–Schulz is compared with the exact polar factor from an SVD. Optimizer state and Muon routing are also checked.
- **Part 2 pipeline**: an evaluator context-window regression test; synthetic end-to-end runs, sweep selection, and Sophia's Hessian refresh inside the training loop.
- **Part 3 decoding**:
  - Cached, left-padded greedy decoding matches uncached, unpadded decoding.
  - Beam width 1 equals greedy.
  - top-k=1 and top-p→0 equal greedy.
  - Exhaustive beam search finds the true maximum-probability sequence.
  - Beam search results don't depend on batching.
  - Banned tokens are never emitted.
  - The top-k and top-p filter definitions are correct.
