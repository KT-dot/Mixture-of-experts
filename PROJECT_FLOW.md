# ANLP Assignment 2: Project Flow and Code Walkthrough

This guide explains what the repository does, how data moves through each part, and what every important function does. It is meant for preparing code explanations. All numbers come from the result files in `assets/results/`, `report/tables/`, `src/part1/shared/data/manifest.json` and the executed Part 1 notebooks.

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Repository map](#2-repository-map)
3. [End-to-end flow](#3-end-to-end-flow)
4. [Part 1: Mixture-of-Experts translation](#4-part-1-mixture-of-experts-translation)
5. [Part 2: Optimizers written from scratch](#5-part-2-optimizers-written-from-scratch)
6. [Part 3: Decoding strategies](#6-part-3-decoding-strategies)
7. [Infrastructure: Ada, SLURM, publishing, report](#7-infrastructure-ada-slurm-publishing-report)
8. [Tests](#8-tests)
9. [Explaining the code: likely questions](#9-explaining-the-code-likely-questions)
10. [Gotchas and loose ends](#10-gotchas-and-loose-ends)
11. [Function index](#11-function-index)

---

## 1. The project in one page

The assignment has three independent experiments. They share one repository, one report (`report.pdf`) and one GPU cluster (IIIT Ada, one RTX 2080 Ti with 11 GB per job).

| Part | Question | Model | Data | Written by hand |
| --- | --- | --- | --- | --- |
| 1 | Does a Mixture-of-Experts (MoE) FFN beat a dense FFN for translation, at equal parameters? | Decoder-only Transformer, d=512, 6 layers, 8 heads, 27.08M params | `belumind/en-vi-ja-curated-500k-triplets` (vi→en and ja→en) | Whole model, 5 FFN variants, router, auxiliary losses, BPE tokenizer, data pipeline, training loop, greedy decoder, BLEU and perplexity evaluation |
| 2 | Which optimizer trains the same LM best on the same token budget? | Part 1's dense model, used as a plain causal LM | `browndw/human-ai-parallel-corpus` | AdamW, NAdamW, Lion, Muon, bonus SophiaG, all as `torch.optim.Optimizer` subclasses; LR sweep; trainer |
| 3 | How do decoding strategies trade quality against diversity? | Pretrained `EleutherAI/pythia-160m`, frozen, FP32 | `hamishivi/ROCStories` test split | Greedy, beam (widths 1/2/4), top-k, top-p, pure sampling, all driven from `model.forward()` with a KV cache; every metric |

The thread that ties the parts together:

- **Part 1** builds the translation Transformer and tests five FFN designs. Every model sees the same batches in the same order, and parameter counts are matched exactly, so differences come from the FFN alone.
- **Part 2** freezes the architecture (the dense variant) and changes only the optimizer. Every run starts from the same saved weights and sees exactly one pass over the data.
- **Part 3** drops training and studies the decoding side, implementing every search and sampling method without `model.generate()`.

Common discipline in all three parts: pinned dataset and model revisions, fixed seeds, hash checks that inputs are identical, every report number read from a result file, and checkpoints published on Hugging Face under `Devatri/anlp-assignment2-*`.

---

## 2. Repository map

```
Assign_2/
├── main.py                     CLI entry point for Parts 2 and 3 (Part 1 is notebook-driven)
├── pyproject.toml, uv.lock     Root uv project (Python 3.12) for Parts 2 and 3
├── configs/
│   ├── part2.yaml              Model, data, training, eval, per-optimizer hyperparameters, sweep grid
│   └── part3.yaml              Pythia revision, ROCStories revision, generation limits, 11 strategies, timing grid
├── src/
│   ├── part1/                  Separate uv project (CUDA 12.4 torch, papermill)
│   │   ├── shared/implementation.ipynb   ALL Part 1 code: model, data, training, eval, routing, checks
│   │   ├── shared/tokenizer_clean.json   Trained 16k byte-level BPE
│   │   ├── shared/data/                  Tokenized splits as .npy + raw.json + manifest.json
│   │   ├── 00_setup_and_checks.ipynb     Data prep, tokenizer training, parameter counts, checks
│   │   ├── model N/model_N.ipynb         One per variant (N=1..5): pilots, full run, eval, routing
│   │   ├── model N/full/checkpoint.pt    Trained weights (git-ignored, on Hugging Face)
│   │   ├── 06_comparison.ipynb           Cross-model tables, fairness asserts, paired bootstrap
│   │   ├── 07_execution_and_upload.ipynb SLURM scripts used on Ada + HF upload cell
│   │   ├── 08_translate_locally.ipynb    Interactive translation demo
│   │   ├── 09_reproducibility_checks.ipynb  Masking, init, cache, resume checks
│   │   └── 10_full_test_bleu.ipynb       Re-scores all checkpoints on the full test split
│   ├── part2/
│   │   ├── config.py           YAML load, validation, deep-merge overrides
│   │   ├── data.py             HF download, schema check, family split, tokenizer, block packing
│   │   ├── model.py            DenseTransformer (Part 1 dense variant, as a causal LM)
│   │   ├── trainer.py          Token-counted training loop, AMP, eval checkpoints, Sophia Hessian refresh
│   │   ├── evaluate.py         Validation loss, continuation BLEU
│   │   ├── experiment.py       Orchestration: shared init, sweep, full runs, metadata
│   │   ├── plotting.py         Loss/BLEU vs tokens, sweep plot
│   │   ├── optimizers/         common.py, adamw.py, nadamw.py, lion.py, muon.py, sophia.py
│   │   └── tests/              7 test files
│   └── part3/
│       ├── config.py           YAML load and validation (requires beam widths 1, 2, 4 and >=1000 samples)
│       ├── decoding.py         IncrementalLM wrapper, greedy, sampling, top-k/top-p filters, beam search
│       ├── metrics.py          NLL/perplexity, token accuracy, bag-of-words P/R/F1, diversity
│       ├── experiment.py       Orchestration, resume fingerprints, beam timing, aggregation
│       ├── plotting.py         Metric bars, quality vs diversity, beam timing
│       └── tests/              test_decoding.py, test_metrics.py
├── scripts/                    SLURM .sbatch files, HF upload, W&B upload, greedy/beam-1 diagnostic
├── assets/results/             Small committed copies of result tables, metadata and plots
├── outputs/                    Full run outputs (git-ignored; locally holds Part 3 outputs)
├── report/                     report.tex, table/figure generators, generated tables and figures
├── report.pdf                  Final report
└── part 1/model 1/model_1.ipynb   Untracked leftover copy (see Section 10)
```

---

## 3. End-to-end flow

```
                 ┌────────────────────────── IIIT Ada cluster (SLURM, RTX 2080 Ti) ──────────────────────────┐
                 │                                                                                            │
 PART 1          │  00_setup_and_checks ──► model 1..5 notebooks (papermill, array job 1-5)                    │
 notebooks       │   prepare_data()          pilot LR {3e-4, 6e-4} on 1M tokens ─► full run 60M tokens         │
                 │   train BPE               test PPL, 1000-triplet BLEU, routing heatmaps                     │
                 │   .npy splits                       │                                                       │
                 │                                     ▼                                                       │
                 │                  06_comparison (fairness asserts, paired bootstrap)                         │
                 │                  10_full_test_bleu (all 24,791 test triplets, unseen-source BLEU)           │
                 │                                                                                            │
 PART 2          │  main.py --part 2 --sweep ─► selected_hyperparameters.json                                  │
 python package  │  main.py --part 2 --hyperparameters ... ─► 5 optimizers × 1 pass, eval every 0.1×           │
                 │                                                                                            │
 PART 3          │  main.py --part 3 ─► 11 strategies (7 sampled × 3 seeds) on 19,611 stories + beam timing    │
                 │  part3_greedy_beam1_diagnostic.py                                                          │
                 └────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                              │
                     checkpoints ─► Hugging Face (scripts/upload_to_hf.py, 07 notebook)
                     logs        ─► Weights & Biases (scripts/upload_logs_to_wandb.py)
                     small results copied ─► assets/results/ and report/tables/
                                                              │
                                                              ▼
                     report/make_report_tables.py ─► report/tex/*.tex
                     report/make_report_figures.py ─► report/figures/report/*.pdf
                     pdflatex report.tex ─► report.pdf
```

How to run each part:

```bash
# Part 1 (inside src/part1, separate uv env)
PART1_PREPARE=1 papermill 00_setup_and_checks.ipynb out.ipynb       # data + tokenizer
PART1_PROFILE=full papermill "model 3/model_3.ipynb" out.ipynb     # one variant (inspect | smoke | full)

# Part 2 (repository root)
python main.py --part 2 --synthetic-smoke                          # CPU pipeline check, no download
python main.py --part 2 --sweep --output outputs/part2/v2          # LR sweep
python main.py --part 2 --hyperparameters outputs/part2/v2/sweep/selected_hyperparameters.json --output outputs/part2/v2

# Part 3
python main.py --part 3                                            # everything, resumes finished runs
python main.py --part 3 --strategies greedy beam_4 --limit 64 --output outputs/p3dbg

# Tests (Parts 2 and 3)
uv run pytest
```

`main.py` parses `--part {2,3}` and dispatches to `run_part2` or `run_part3`. Imports are lazy (inside the functions), so Part 3 never imports Part 2 code and vice versa. `--synthetic-smoke` shrinks the Part 2 model to d=32, 1 layer, sequence length 8, on CPU in float32, using a synthetic modular-pattern dataset.

---

## 4. Part 1: Mixture-of-Experts translation

### 4.1 Flow

```
HF dataset (pinned revision 849990da…)
   │  prepare_data()                    [implementation.ipynb cell 5]
   ▼
NFC normalise → overlap/duplicate audit → train 16k byte-level BPE on 50k train triplets
   │
   ▼
each triplet → 2 sequences: [BOS] vi [SEP] en [EOS]  and  [BOS] ja [SEP] en [EOS]
drop triplet if either direction > 256 tokens (never truncate)
   │  saved as flat.npy / offsets.npy / seps.npy / langs.npy / rows.npy + raw.json
   ▼
PairStore (memory-mapped) → epoch_batches (length-bucketed) → schedule (whole batches up to 60M tokens)
   │
   ▼
model N notebook:  run_checks → 2 pilot runs (1M tokens) → pick LR on val NLL → training_run (60M tokens)
   │
   ▼
evaluate_nll (val, test) → evaluate_bleu (1,000 triplets, bootstrap) → routing_diagnostics → result JSON
   │
   ▼
06_comparison (tables, asserts, paired bootstrap)   10_full_test_bleu (full test split)
```

### 4.2 Notebook order

All code lives in `src/part1/shared/implementation.ipynb`. Code cells tagged `definitions` hold the functions. Every other notebook starts with the same loader cell: it finds the folder that contains `shared/implementation.ipynb`, reads that notebook as JSON, and `exec`s each `definitions` cell into its own globals. This keeps one source of truth without a Python module.

| Notebook | What it does |
| --- | --- |
| `00_setup_and_checks.ipynb` | Prints versions, asserts parameter counts (M1–M4 equal totals; M5 active = M1 total), runs `run_checks` for all 5 variants, and runs `prepare_data()` when `PART1_PREPARE=1`. |
| `model N/model_N.ipynb` | `PART1_PROFILE` = `inspect` (checks only, default), `smoke` (tiny model, labelled "SMOKE ONLY") or `full`. Full: two 1M-token pilots, choose LR, train 60M tokens (or load `checkpoint.pt`), evaluate, plot, and emit one JSON result record in a cell tagged `part1-result`. |
| `06_comparison.ipynb` | Reads the `part1-result` JSON from each executed model notebook (no hand-typed scores). Asserts identical schedule digest, token counts, tokenizer hash, dataset revision and BLEU sample hash. Plots PPL/BLEU, runs a paired bootstrap of each model's BLEU against M1. |
| `10_full_test_bleu.ipynb` | Loads each `full/checkpoint.pt`, translates all 24,791 test triplets (49,582 translations), computes BLEU with bootstrap CI, and BLEU on test triplets whose sources never appear verbatim in train. Writes `full_test_bleu.json`. |
| `07_execution_and_upload.ipynb` | Holds the exact SLURM scripts used on Ada (prepare job 2711534, training array 2711535), plus an opt-in Hugging Face upload cell. |
| `08_translate_locally.ipynb` | Loads the checkpoint with the lowest validation NLL (or a chosen one) and translates user sentences on CPU. |
| `09_reproducibility_checks.ipynb` | Real-data masking check, identical backbone init across variants, cache with unequal prompt lengths, and bitwise-identical resume. |

### 4.3 Data pipeline: `prepare_data()` (cell 5)

1. Load the dataset at pinned revision `849990daee76e0f9e2eb9965e30e34bc1909a93d`.
2. `canonical(text)` = Unicode NFC normalisation + strip.
3. **Leakage audit.** Drop training rows whose English target appears in validation or test, drop validation rows whose target appears in test, and drop exact duplicate training triplets. On this dataset the audit removed **0 rows** in every category (see `manifest.json`).
4. **Tokenizer.** `tokenizers` BPE with NFC normaliser, ByteLevel pre-tokenizer and decoder, vocab 16,000, `min_frequency=2`, special tokens `[PAD]=0, [BOS]=1, [EOS]=2, [SEP]=3`, and the full 256-byte initial alphabet (so no unknown tokens). Trained on 50,000 shuffled training triplets (seed 42), all three languages. Round-trip decode is asserted on 1,000 validation sentences per language.
5. **Encoding.** For each triplet, build `ids = [BOS, *src, SEP, *tgt, EOS]` for vi and ja. Drop the direction if it contains a reserved id, is longer than 257 ids (256 inputs after shifting), or is empty. If either direction is dropped, the whole triplet is dropped, so vi and ja always share the same reference rows.
6. **Storage** per split (`shared/data/<split>/`):

| File | Content |
| --- | --- |
| `flat.npy` | All sequences concatenated (int32) |
| `offsets.npy` | Prefix sums of sequence lengths (int64); sequence `i` = `flat[offsets[i]:offsets[i+1]]` |
| `seps.npy` | Index of the target start (= len(src) + 1) |
| `langs.npy` | 0 = vi, 1 = ja (int8) |
| `rows.npy` | Triplet row id (links both directions to one triplet) |
| `raw.json` | Retained texts (en, vi, ja, quality score, corpus source) for BLEU references |

Result: **446,202 / 24,789 / 24,791** triplets (train/val/test), i.e. 892,404 / 49,578 / 49,582 sequences. Only 50 + 3 + 1 triplets were dropped, all for length. Training split: 38,449,379 input tokens and 18,737,614 target tokens per epoch.

`PairStore(split)` memory-maps the `.npy` files and exposes `lengths` (input length = sequence length − 1) and `get(i)`.

### 4.4 Sequence format and label masking: `collate()`

Worked example with a 2-token source and 2-token target:

```
ids   = [BOS, s1, s2, SEP, e1, e2, EOS]      sep = len(src)+1 = 3,  n = len(ids)-1 = 6
x     = [BOS, s1, s2, SEP, e1, e2]           ids[:-1]          (model input)
y     = [-100,-100,-100, e1, e2, EOS]         y[sep:n] = ids[sep+1:]
```

- The model input at `SEP` predicts the first English token. Source positions carry `-100`, so the loss covers only English tokens and EOS.
- Batches are **right-padded** with `PAD=0`; padded labels stay `-100`.
- The source language is never given to the model; it must infer it from the script.

### 4.5 Batching and the identical schedule

`epoch_batches(ds, cfg, epoch)`:

1. Shuffle all indices with `rng = default_rng(seed + epoch)`.
2. Take pools of 4,096 examples, sort each pool by length (stable).
3. Greedily pack a batch until adding one more would exceed `(len(batch)+1) × longest > 8192` padded tokens or 128 examples.
4. Shuffle the order of the batches.

`schedule(ds, cfg, budget)` concatenates epochs and appends **whole** batches until the next one would exceed the 60M token budget. Realised: **12,473 batches, 59,998,213 non-padding input tokens (about 1.56 epochs), 29,239,313 supervised target tokens.** Batching depends only on the seed and data, never on the model, so all five variants get the same batches. `schedule_digest()` hashes the batch index arrays; `06_comparison` asserts the digest is identical across models, and resume asserts it matches.

### 4.6 The model

`Config` (dataclass): `vocab=16000, d=512, heads=8, layers=6, hidden=2048, context=256, dropout=0.1, seed=42, token_budget=60M, batch_tokens=8192, batch_examples=128, lr=6e-4, weight_decay=0.1, balance_coef=0.01, z_coef=0.001, eval_interval=5M, val_examples=2048, max_new_tokens=160`.

**`Attention`**

- One fused `qkv` linear (512 → 1536, no bias) reshaped to `(3, batch, heads, time, 64)`.
- **Rotary embeddings (RoPE)**: cos/sin tables for 256 positions with frequencies `10000^(-2i/64)`. `rotate()` pairs even and odd channels `(a, b)` and maps them to `(a·cos − b·sin, a·sin + b·cos)`. Positions are passed explicitly, which is what lets cached decoding put a new token at the right position.
- Training and prefill: `F.scaled_dot_product_attention(..., is_causal=True)`. Right padding is safe here because no real token can see a later padding token.
- Cached step: concatenate cached K/V, and use `attn_mask=valid` instead of a causal mask (a single new query may attend to every earlier real token).
- Returns the output and the new `(k, v)` cache.

**`Block`** (pre-LayerNorm): `x = x + drop(attn(LN1(x)))`, then `x = x + drop(ffn(LN2(x)))`. The FFN also returns the balance loss and z-loss. `local_valid = valid[:, -x.shape[1]:]` trims the mask during cached steps, where `x` holds one token but `valid` covers the whole history.

**`Transformer`**

- Embedding with `padding_idx=PAD`, 6 blocks, final LayerNorm, embedding dropout.
- **Tied output**: logits are `F.linear(hidden, embedding.weight)`; there is no separate LM head.
- **Initialisation**: every Linear/Embedding is drawn from N(0, 0.02). Then every non-FFN matrix is re-drawn with its own generator seeded by `seed + sha256(parameter name)`. Variants create different numbers of FFN parameters, which would shift the global RNG; per-name seeding makes the **shared backbone bitwise identical across all five variants** (checked in `09`). The PAD embedding row is zeroed.
- **Forward with labels** (training/eval): logits are computed only at supervised positions (`hidden[mask]`), which saves memory, and cross-entropy runs in float32. It returns per-token losses, the mean balance loss and the mean z-loss over layers, and the mask.
- **Forward without labels** (generation): returns full logits and the per-layer KV cache.

`parameter_counts(cfg)` builds the model on the `meta` device (no memory) and returns total parameters and min/max active parameters per token.

### 4.7 The FFN / MoE layer: `FFN.forward(x, valid)`

Dense (variant 1): `down(gelu(up(x)))`, 512 → 2048 → 512, no bias. Returns zero auxiliary losses.

MoE (variants 2–5), step by step:

1. Flatten to `(tokens, d)` and keep only non-padding positions (`valid`). Padding never reaches the router or the auxiliary losses.
2. **Router logits** in float32: `logits = tokens @ W_router^T` (`F.linear(tokens.float(), router.weight.float())`). Float32 keeps the softmax and z-loss stable under FP16 autocast.
3. `probs = softmax(logits)` over **all** experts, then `topk`.
4. Weights: top-2 renormalises the two selected probabilities to sum to 1. Top-1 keeps the **raw** softmax probability. If it were renormalised, it would always be 1 and the router would get no gradient from the task loss.
5. **Sparse dispatch**: for each expert `e`, `torch.where(indices == e)` gives the (token, slot) pairs routed to it. The expert runs only on those tokens, the outputs are scaled by their weights, and `index_add_` adds them into `mixed`. There is no capacity limit, so no token is dropped.
6. Variant 4 only: `experts[0]` is the **shared** expert, applied to every token and scaled by `sigmoid(shared_gate(token))`. The router has 3 outputs indexing `experts[1:]` (hence `offset = 1`).
7. Scatter `mixed` back to the full `(batch × time)` grid with `index_copy_`.
8. Auxiliary losses (Section 4.9).
9. If `self.capture` is set, store `(positions, expert indices, weights)` for routing diagnostics.

`run_checks` verifies that this sparse dispatch equals a dense, independently assembled mixture to 1e-6.

### 4.8 The five variants and parameter matching

| Variant | Experts (hidden widths) | Router | Top-k | Total params | Active params per token |
| --- | --- | --- | --- | --- | --- |
| M1 dense | one MLP, 2048 | none | – | 27,079,680 | 27,079,680 |
| M2 | 4 × [512, 512, 511, 511] | 512 → 4 | 1 | 27,079,680 | 17.649M–17.655M |
| M3 | 4 × [512, 512, 511, 511] | 512 → 4 | 2 | 27,079,680 | 20.788M–20.801M |
| M4 | shared 512 + 3 routed [512, 511, 511] | 512 → 3, plus 512 → 1 sigmoid gate | 1 routed + shared | 27,079,680 | 20.794M–20.801M |
| M5 | 4 × 1023 | 512 → 4 | 2 | 39,650,304 | 27,079,680 |

The arithmetic, per layer:

- Dense FFN: 2 × 512 × 2048 = **2,097,152**.
- M2–M4: the router costs 4 × 512 = 2,048 (M4: 3 × 512 + 512 for the gate). The remaining 2,095,104 = 2 × 512 × 2,046, so the experts share **2,046 hidden units**: `divmod(2046, 4)` gives widths [512, 512, 511, 511]. Each layer is exactly 2,097,152, so the totals match M1.
- M5: four experts of width 1,023. Two active experts plus the router = 2 × 512 × 2,046 + 2,048 = 2,097,152, so **active** parameters equal M1's total. The model total is 39,650,304.
- Active counts in M2–M4 are ranges because a token may land on a 511- or 512-wide expert.

Full model check: embedding 16,000 × 512 = 8,192,000. Per layer: qkv 786,432 + out 262,144 + 2 LayerNorms 2,048 + FFN 2,097,152 = 3,147,776; × 6 = 18,886,656. Final LayerNorm 1,024. Total **27,079,680**.

### 4.9 Auxiliary losses

Both are averaged over the 6 layers and added to the task loss: `loss = mean(token NLL) + 0.01 × balance + 0.001 × z`.

- **Load balancing (Switch Transformer style)**: `balance = E × Σ_e f_e × P̄_e`. `f_e` is the fraction of routing assignments that went to expert `e` (one-hot of the top-k indices, divided by k, **detached**). `P̄_e` is the mean router probability of `e`. The gradient flows through `P̄_e` only. The minimum, 1.0, is reached when routing is uniform. Final values were 1.01–1.08.
- **Router z-loss**: `mean(logsumexp(logits)²)`. It keeps router logits small so the softmax stays numerically stable.
- Neither term enters perplexity.

### 4.10 Training loop: `training_run(cfg, train, val, out, budget)` (cell 6)

1. Build the schedule and its digest; `seed_all(42)`; build the model.
2. **Optimizer**: `torch.optim.AdamW` (Part 1 is not about optimizers) with betas (0.9, 0.95), eps 1e-8. Weight decay 0.1 applies to matrices only (`ndim ≥ 2`); LayerNorm parameters get 0.
3. **LR schedule**, with `progress = tokens done / realised budget`:
   - warm-up over the first 3%: `factor = max(0.01, progress / 0.03)`;
   - then cosine decay from 1.0 to 0.1: `0.1 + 0.9 × 0.5 × (1 + cos(π × (progress − 0.03) / 0.97))`.
4. **FP16 autocast** with `GradScaler(init_scale=128)`: scale, backward, `unscale_`, `clip_grad_norm_(1.0)`, step, update.
5. If the scaler ever **skips** a step (its scale dropped), the run raises `FloatingPointError`. Every variant must take exactly 12,473 real updates; this never triggered.
6. A non-finite loss also raises.
7. Log training NLL every 100 steps. Every 5M tokens, run `evaluate_nll` on a fixed 2,048-pair validation subset (seed 109) and save `resume.pt` (model, optimizer, scaler, RNG states, history, counters). That is 12 validation points per run.
8. **Resume**: if `resume.pt` exists, assert config and schedule digest match, restore everything including RNG states, and continue. `09` proves an interrupted and an uninterrupted run end with bitwise-identical weights.
9. Save `checkpoint.pt` = model, config, summary (tokens, steps, digest, seconds, last validation), history, tokenizer hash. Writes are atomic (save to `.tmp`, then rename).

**LR pilots** (in each model notebook): train from scratch for 1M tokens at 3e-4 and at 6e-4 and pick the lower validation NLL. All five picked **6e-4**, the upper edge of the grid (pilot NLLs 6.39–6.46 vs 6.59–6.67).

### 4.11 Evaluation

**`evaluate_nll(model, ds, cfg, indices)`**: teacher-forced, batched with `epoch_batches(shuffle=False)`, sums NLL per language in float64. Returns `nll`, `ppl = exp(total NLL / target tokens)`, and per-language NLL/PPL. Test PPL covers all 49,582 test pairs (1,038,360 target tokens).

**`greedy_translations(model, tok, texts, cfg)`**:

1. Encode sources and build prompts `[BOS, *src, SEP]` (asserting they fit the context; nothing is truncated).
2. **Right-pad** the batch and run one prefill forward to get logits and the KV cache. Take the logits at each row's last real position (`lens − 1`).
3. Each step: set PAD/BOS/SEP logits to −inf, take the argmax, stop a row at EOS, at 160 new tokens, or at the context limit (counted as "capped").
4. Append the new token **after** the padding. `valid` grows by one True column, so the gap of padding between the prompt and the new tokens stays masked. Positions are `lens + t`, so RoPE sees the correct position despite the gap. `09` checks cached logits against uncached ones with unequal prompt lengths.
5. Decode with the BPE byte-level decoder.

**`evaluate_bleu(model, tok, cfg, nrows=1000, seed=2026, bootstrap=300)`**:

- Fixed sample of 1,000 test triplets (seed 2026; its index hash is recorded so all models share it).
- Translate vi→en and ja→en; sacreBLEU corpus BLEU per language and combined. Signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.
- **Paired triplet bootstrap**: resample triplets with replacement and keep both translations of the same triplet together; the 95% CI is the 2.5–97.5 percentiles of 300 resamples.
- `10_full_test_bleu.ipynb` reuses this function with `nrows = all 24,791` and 200 resamples.

### 4.12 Routing diagnostics: `routing_diagnostics(model, ds, cfg)` (cell 7)

- Turn on `ffn.capture` in every block and run 1,000 test pairs (seed 909) teacher-forced.
- Split tokens into two regions: **source** (positions between BOS and SEP) and **English prefix** (positions after SEP).
- Count, per layer × language × region × expert, how often each expert is selected. Rates are counts / tokens in the region.
- Assert rates sum to k per token (1 for M2; 2 for M3, M5, and M4, where the shared expert counts as expert 0 at 100%).
- Per cell: normalised routing entropy `H = −Σ p log p / log(#routed experts)` (1 = uniform), number of unused experts, and the largest share.
- `plot_model_results` draws the training/validation curves and a heatmap per layer and region (rows vi/ja, columns E0–E3).

### 4.13 Correctness checks

`run_checks(variant)` runs on a tiny model (d=32, 2 layers, vocab 96) before every real run:

| Check | How |
| --- | --- |
| Causal attention | Changing a future token leaves earlier logits unchanged |
| Padding | A padded row's real logits equal the same row run alone |
| KV cache | Incremental logits equal the full-forward logits |
| Target loss | Masked loss equals direct cross-entropy on the masked logits |
| Router gradient | Every router weight gets a finite, non-zero gradient |
| Sparse dispatch | Equals a dense mixture computed independently |
| Tiny overfit | 35 AdamW steps reduce loss below 60% of the initial value |
| Checkpoint round trip | Save/load gives identical logits |

`09_reproducibility_checks.ipynb` adds the real-data masking check, the identical-backbone check, the unequal-length cache check and the bitwise resume check.

### 4.14 Results (full test split)

| M | FFN | Active | Test PPL | PPL vi | PPL ja | BLEU | 95% CI | vi→en | ja→en | Unseen | Train min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Dense | 27.08M | 5.245 | 4.451 | 6.180 | 33.98 | 33.71–34.31 | 39.07 | 28.79 | 33.56 | 11.6 |
| 2 | 4E top-1 | 17.65M | 5.356 | 4.567 | 6.280 | 33.51 | 33.26–33.87 | 38.50 | 28.43 | 33.09 | 17.6 |
| 3 | 4E top-2 | 20.79M | 5.339 | 4.523 | 6.301 | 33.56 | 33.30–33.91 | 38.62 | 28.40 | 33.13 | 17.1 |
| 4 | Shared + 3E top-1 | 20.80M | 5.380 | 4.560 | 6.349 | 33.60 | 33.34–33.93 | 38.65 | 28.45 | 33.19 | 17.1 |
| 5 | 4E top-2 (matched active) | 27.08M | **5.165** | 4.387 | 6.080 | **34.13** | 33.85–34.47 | 39.03 | 29.14 | 33.69 | 17.5 |

What the numbers say:

- **PPL order: M5 < M1 < M3 < M2 < M4**, the same for both languages. Perplexity is the reliable signal (about a million target tokens).
- At equal **total** parameters (M2–M4), MoE is 1.8–2.6% worse in PPL but uses 23–35% fewer parameters per token.
- At equal **active** parameters (M5), extra inactive capacity gives the best model.
- BLEU differences are inside the bootstrap CIs. Paired bootstrap vs M1 on 1,000 triplets: every interval contains zero.
- ja→en is much harder than vi→en (BLEU about 28.5 vs 39; PPL about 6.2 vs 4.5).
- MoE trains slower (17 vs 11.6 minutes) despite fewer FLOPs: four small gathered matmuls use the GPU less efficiently than one 2048-wide matmul.
- **Routing**: experts specialise by **token role** (source vs English), not by language. vi and ja columns are nearly identical. Early source-side layers are the most imbalanced (M2 layer-1 entropy 0.11–0.18), deeper layers are near uniform, and no expert ever goes unused.

The model notebooks' headline BLEU (for example M1 35.13) is on the 1,000-triplet sample. The report uses the full-test numbers above.

---

## 5. Part 2: Optimizers written from scratch

### 5.1 Flow

```
main.py --part 2
   │  load_config(configs/part2.yaml) → validate_config
   │  (--hyperparameters) apply_selected_hyperparameters → overlay sweep LRs
   ▼
_prepare_data_and_initial_state()                         [experiment.py:108]
   │  prepare_real_data(): resolve revision → load → inspect schema → choose text/id fields
   │                       → split 90/5/5 by document family → train BPE on train
   │                       → pack <bos> doc <eos> stream into 257-token rows
   │  seed → build DenseTransformer → save initial_state.pt + SHA-256
   ▼
for each optimizer (adamw, nadamw, lion, muon, sophia):
   │  _fresh_model_and_optimizer(): reseed, load initial state, assert hash, make_optimizer
   │  train_optimizer(): fixed order, warm-up + cosine, FP16 + GradScaler,
   │                     at every 0.1× → val loss + test BLEU (+ Sophia Hessian refresh every 10 batches)
   │  _write_tables(): metrics.csv/json, generations.csv, plots (after every optimizer)
   ▼
run_metadata.json, resolved_config.yaml, muon_parameter_routing.json, checkpoints/<opt>/fraction_1.0.pt
```

`run_sweep()` uses the same setup, but each trial trains on only 0.1× of the data with evaluation disabled except the final validation loss.

### 5.2 Config (`configs/part2.yaml`, `src/part2/config.py`)

- `validate_config` checks that all sections exist, evaluation fractions are unique, increasing, in (0, 1] and end at 1.0, optimizers are known and not duplicated, and each has hyperparameters.
- `with_overrides(config, overrides)` deep-merges a dict into a copy and re-validates. It is used for smoke tests, sweep trials and sweep-selected LRs.
- Base LRs (later replaced by the sweep): AdamW/NAdamW 6e-4, Lion 1e-4, Muon 0.02 (fallback AdamW 6e-4), Sophia 3e-4.

### 5.3 Data (`src/part2/data.py`)

1. **Pinned revision.** `HfApi().dataset_info(...).sha` resolves the configured revision `b514ff64…` to an immutable commit before download.
2. **Schema inspection before field choice.** `inspect_dataset_schema` records columns and types. `choose_text_and_id_fields` picks `text` (or the only string column) and `doc_id`, and fails instead of guessing if it is ambiguous.
3. **Split by document family.** The dataset has only a train split. Its 66,320 rows are human- and AI-written versions of the same source documents. `_document_family(doc_id)` takes the part before `@`. Families are shuffled with seed 42 and split 90/5/5, giving **59,688 / 3,312 / 3,320 documents**. Parallel versions of one document never cross splits, which would otherwise leak near-duplicate text into test.
4. **Tokenizer.** `train_or_load_tokenizer`: byte-level BPE, 16,000 vocab, `min_frequency=2`, special tokens `<pad>, <bos>, <eos>, <unk>`, NFKC normaliser, 256-byte initial alphabet. Trained on the training split only. A `tokenizer_manifest.json` (dataset, revision, split fingerprint, settings) must match before a cached tokenizer is reused. Result: 0 unknown tokens in every split.
5. **Packing.** `texts_to_blocks`: build one long stream `<bos> doc <eos> <bos> doc <eos> …`, then `unfold(0, 257, 256)` into overlapping rows of 257 tokens with stride 256. Row `r` gives inputs `row[:-1]` and labels `row[1:]`, so every token is predicted exactly once. There is no padding.
6. Result: **153,169 / 8,523 / 8,510 blocks**. One pass = 153,169 × 256 = **39,211,264 training tokens** (the "1× dataset" budget).
7. `fixed_training_order(n, seed)` = one `randperm` with seed 42, shared by every optimizer.

### 5.4 Model (`src/part2/model.py`)

`DenseTransformer` is Part 1's dense variant rewritten as a clean module: d=512, 8 heads, 6 pre-LN blocks, GELU MLP 2048, RoPE, dropout 0.1, no biases, tied embedding (`lm_head.weight = embedding.weight`), 27,079,680 parameters.

Differences from Part 1:

| | Part 1 `Transformer` | Part 2 `DenseTransformer` |
| --- | --- | --- |
| Input | `[BOS] src [SEP] tgt [EOS]`, right-padded | Packed 256-token blocks, no padding |
| Loss | Only English/EOS positions | Every position (`ignore_index=-100` unused in practice) |
| RoPE positions | Explicit `positions` tensor | Offset = cached length |
| Cache step | Masked by `valid` | One token at a time, no mask needed |
| Tied head | `F.linear(hidden, embedding.weight)` | Separate `lm_head` sharing the embedding weight |
| Init | Per-name seeded backbone | N(0, 0.02) everywhere |

`greedy_generate(prompt, n)` runs a prefill with `use_cache=True` and then `n` single-token steps. It always generates exactly `n` tokens (no EOS stop).

### 5.5 Fairness controls

- **Same initial weights.** `_prepare_data_and_initial_state` seeds, builds the model once, saves `initial_state.pt`, and computes `state_dict_sha256` (hash of names, dtypes, shapes and bytes). `_fresh_model_and_optimizer` reloads that state for every optimizer and **asserts the hash matches**.
- **Same data order**, same batch size (32 × 256 = 8,192 tokens), same schedule shape, same clipping, same precision.
- **Only the optimizer and its learning rate change.**
- `seed_everything` sets the Python/NumPy/Torch seeds, `torch.use_deterministic_algorithms(True, warn_only=True)` and cuDNN deterministic mode; the SLURM script sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

### 5.6 Training loop: `train_optimizer()` (`trainer.py:97`)

1. Budget: `token_budget = None` means one full pass. `example_budget = budget / 256`, and the order is the first `example_budget` indices of the fixed permutation.
2. Targets: `ceil(example_budget × f)` for f = 0.1, 0.2, …, 1.0. The inner loop takes `min(32, target − processed)` examples, so a batch never crosses an evaluation point. That gives 4,790 batches per pass instead of 4,787.
3. **LR**: `learning_rate_scale(progress, 0.05, 0.1)` is linear warm-up over 5% of tokens, then cosine from 1.0 to 0.1 of the peak. `set_learning_rate` stores each group's `base_lr` once and multiplies it, which handles Muon's two groups with different LRs.
4. **FP16** (the 2080 Ti has no native BF16): `scaler.scale(loss).backward()` then `scaler.step(optimizer)`. `GradScaler.step` unscales the gradients first, so the optimizer's own clipping sees true gradient values. If gradients overflow, the scaler skips the step and lowers its scale; the loop detects the skip as `scale_after < scale_before` (`trainer.py:176`) and counts it. The batch still counts as data seen. AdamW and NAdamW each had 1 skip; the others had 0.
5. **Sophia Hessian refresh** every 10 batches (`trainer.py:185`), see 5.8.
6. At each evaluation point: compute pure training time (total minus earlier evaluation time), validation loss and perplexity on all 8,523 validation blocks, test BLEU on all 8,510 test blocks, and Sophia's clipped fraction. Append to `metrics` and `progress_<opt>.jsonl`.
7. Save `checkpoints/<opt>/fraction_1.0.pt` (model, model config, optimizer state, scaler, counters, initial-state hash) at the end of the pass only (quota).
8. Assert that exactly `token_budget` tokens were processed.

### 5.7 Evaluation (`evaluate.py`)

- `precision_context(device, precision)` returns an autocast context (FP16 on CUDA here).
- `evaluate_loss`: token-weighted mean cross-entropy over the split; perplexity = `exp(min(loss, 80))`.
- `evaluate_continuation_bleu`: for every test block, the prompt is the first 128 tokens (`prefix_fraction 0.5`). The model greedily generates 64 tokens and sacreBLEU compares them with the **true next 64 tokens**. One corpus BLEU over all 8,510 blocks. `generations.csv` keeps all examples at the final checkpoint and 64 at earlier ones. Absolute BLEU is low (below 4) because open-ended continuations have many valid wordings; it ranks optimizers, it does not measure quality on its own.

### 5.8 The optimizers

All five subclass only `torch.optim.Optimizer`; no built-in update rule is called. Each `step()`:

1. computes one global clipping scale `min(1, 1.0 / ‖g‖₂)` over all groups (`common.global_gradient_scale`; all groups must share one `max_grad_norm`);
2. casts each gradient to float32 and scales it (`dense_float_gradient`);
3. keeps state tensors in float32;
4. applies **decoupled weight decay** `w ← (1 − ηλ) w` (`common.decoupled_decay`), then the update `w ← w − η · u`.

Parameters with no gradient are skipped entirely (no decay either). In Part 2 every optimizer receives all parameters in one group (Muon: two groups), so decay also applies to LayerNorm parameters, identically for every optimizer.

**AdamW** (`adamw.py`, Algorithm 1 of arXiv:2509.02046):

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\hat m = m_t/(1-\beta_1^t), \quad \hat v = v_t/(1-\beta_2^t), \quad w \leftarrow (1-\eta\lambda)w - \eta\, \hat m/(\sqrt{\hat v}+\epsilon)
```

Betas (0.9, 0.95), eps 1e-8, λ = 0.1. State: `step`, `exp_avg`, `exp_avg_sq`.

**NAdamW** (`nadamw.py`, Algorithm 2): the same moments, but the update uses the **Nesterov look-ahead** `m̃ = β₁ m_t + (1 − β₁) g` (`nadamw.py:76`), bias-corrected by `1 − β₁^t`. The momentum is applied one step early. A test checks that it differs from AdamW.

**Lion** (`lion.py`, arXiv:2302.06675):

```math
u = \mathrm{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t), \quad w \leftarrow (1-\eta\lambda)w - \eta u, \quad m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t
```

Betas (0.9, 0.99), λ = 0.3. The update interpolates with β₁ but the momentum is tracked with β₂. Every coordinate moves by exactly ±η, which is why Lion needs a smaller LR and larger λ. State: one tensor (`exp_avg`), half of Adam's (108 MB vs 217 MB here).

**Muon** (`muon.py`, Algorithm 8 + Keller Jordan's reference):

- **Routing** (`is_muon_parameter`): 2-D tensors whose name starts with `blocks.` and ends with `.weight`, i.e. the 24 matrices `qkv`, `out`, `up`, `down` in 6 blocks (18.87M parameters). Everything else (tied embedding/LM head, LayerNorm weights and biases, 27 tensors, 8.21M parameters) goes to a hand-written AdamW fallback with LR 6e-4. `build_muon_parameter_groups` makes the two groups with an explicit `use_muon` flag; the routing table is saved to `muon_parameter_routing.json`.
- **Matrix update** (`muon.py:198`):

```math
m_t = \beta m_{t-1} + g_t, \quad u = \beta m_t + g_t, \quad w \leftarrow (1-\eta\lambda)w - \eta \sqrt{\max(1, r/c)}\; \mathrm{NS}_5(u)
```

  Momentum 0.95. Note the momentum has no `(1 − β)` factor; the Newton–Schulz step normalises scale anyway.
- **Newton–Schulz** (`newton_schulz_orthogonalize`): normalise `X = u / (‖u‖_F + eps)`, transpose if tall (so `X Xᵀ` is the smaller Gram matrix), then five iterations of `X ← 3.4445 X + (−4.7750 A + 2.0315 A²) X` with `A = X Xᵀ`, in float32. This pushes all singular values towards 1, approximating the polar factor `U Vᵀ` of `u = U Σ Vᵀ`. The update therefore has the same "size" in every direction. A test compares it with the exact SVD polar factor.
- **Shape scaling**: `sqrt(max(1, rows/cols))` with PyTorch's `(out, in)` weight layout gives ×2 for `up` (2048×512), ×√3 for `qkv` (1536×512), ×1 for `out` and `down`.

**SophiaG** (`sophia.py`, bonus, arXiv:2305.14342):

- Each step: `m ← β₁ m + (1 − β₁) g`, then `w ← (1 − ηλ) w − η · sign(m) · min(|m| / (ρ · B · h + ε), 1)` (`sophia.py:122`). Betas (0.965, 0.99), ρ = 0.04, ε = 1e-15, B = tokens per batch = 8,192.
- **Hessian estimate** (`trainer._sampled_label_hessian_update`, Gauss–Newton–Bartlett): every 10th batch, run the model on the same inputs, **sample labels from the model's own softmax**, back-propagate the cross-entropy on those sampled labels, and call `optimizer.update_hessian()`: `h ← β₂ h + (1 − β₂) ĝ²`. The squared gradient of a loss on self-sampled labels is an unbiased estimate of the Gauss–Newton diagonal.
- Under FP16 the sampled loss is multiplied by the current GradScaler scale and divided back in `update_hessian(gradient_divisor=scale)`. A non-finite estimate is discarded and counted as rejected (none were).
- `B` rescales `h` because the sampled loss is a per-token mean.
- `clipped_fraction()` reports the share of coordinates where the ratio hit 1: **93.0–94.2%** throughout, so Sophia behaved mostly like signed momentum.

State memory per parameter (float32):

| Optimizer | State tensors |
| --- | --- |
| AdamW, NAdamW | `exp_avg`, `exp_avg_sq` (2×) |
| Lion | `exp_avg` (1×) |
| Muon | matrices: `momentum_buffer` (1×); fallback: 2× |
| Sophia | `exp_avg`, `hessian` (2×) |

### 5.9 Learning-rate sweep: `run_sweep()` (`experiment.py:174`)

- Each trial trains 0.1× of one pass (3,920,896 tokens) from the shared initial weights, with **its own complete warm-up + cosine schedule**, and is scored on **validation loss** only. The test set is never used for selection.
- Grid: multipliers {¼, ½, 1, 2, 4} × base LR. If the best trial is on an edge, the grid extends ×2 in that direction (up to 3 times) until the minimum is interior.
- Resumable: finished trials in `sweep.csv` with the same optimizer, LR and budget are reused. `best_trial` only considers multipliers on the current grid, so a resumed sweep replays the same extension decisions.
- Writes `selected_hyperparameters.json` and `sweep_validation_loss_vs_lr.png`.

| Optimizer | Base LR | Selected LR (multiplier) | Proxy val loss | Extensions |
| --- | --- | --- | --- | --- |
| AdamW | 6e-4 | 1.2e-3 (2×) | 5.430 | 0 |
| NAdamW | 6e-4 | 2.4e-3 (4×) | 5.371 | 1 (to 8×) |
| Lion | 1e-4 | 4e-4 (4×) | 5.860 | 1 (to 8×) |
| Muon | 0.02 | 0.04 (2×) | 5.086 | 0 |
| Sophia | 3e-4 | 3e-4 (1×) | 5.792 | 0 |

Only the peak LR is swept. Muon's fallback AdamW LR (6e-4) is not.

### 5.10 Results (one full pass)

| Optimizer | LR | Val loss 0.5× | Val loss 1.0× | Val PPL | BLEU 0.5× | BLEU 1.0× | Train s | Peak GiB | AMP skips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AdamW | 1.2e-3 | 4.179 | 3.849 | 47.0 | 2.93 | 3.78 | 370 | 4.69 | 1 |
| NAdamW | 2.4e-3 | 4.179 | **3.824** | 45.8 | 3.01 | **3.89** | 390 | 4.69 | 1 |
| Lion | 4e-4 | 4.777 | 4.374 | 79.3 | 1.52 | 2.07 | 359 | 4.59 | 0 |
| Muon | 0.04 | 4.392 | 3.913 | 50.1 | 2.30 | 3.70 | 631 | 4.62 | 0 |
| Sophia | 3e-4 | 4.559 | 4.157 | 63.9 | 1.80 | 2.56 | 414 | 4.92 | 0 |

- Final order **NAdamW < AdamW < Muon < Sophia < Lion** in validation loss, exactly reversed in BLEU.
- **Muon leads early** (lowest loss at 0.1×: 5.184) and in the sweep, then the Adam family overtakes it from 0.2×. Likely causes: 30% of parameters (including the big tied embedding) use an unswept fallback LR, and the budget is only about 1.4 tokens per parameter. It is also the slowest (+70% time) because of the five FP32 Newton–Schulz iterations per matrix per step.
- **Lion** suffers from the short-proxy sweep. Extra full-pass runs (`assets/results/part2/lion_lr_check/`) reached 4.297 at 1e-4 and 4.256 at 2e-4, both better than the selected 4e-4.
- **Sophia**: ~94% of coordinates clipped, so the update is mostly `η·sign(m)`; the Hessian pass adds 12% time and the highest memory.

---

## 6. Part 3: Decoding strategies

### 6.1 Flow

```
main.py --part 3 → load_config(configs/part3.yaml) → run_experiment()        [experiment.py:431]
   │  seed 1234; TF32 off; load Pythia-160M (pinned, FP32); model.generate = None
   │  prepare_stories(): 19,633 test rows → drop 22 with empty continuation → 19,611
   │  build_masks(): stop = {EOS}; banned = 188 newline-containing token ids
   │  IncrementalLM(model, pad=EOS, vocab=50,277, banned_mask)
   │  _reference_scores(): perplexity of the human continuations (24.71)
   ▼
for each strategy × seed (skip if runs/<name>.json has the same run fingerprint):
   │  _decode_batches()  → greedy_decode | beam_search_decode | sample_decode   (CUDA-synchronised timing)
   │  _score_run()       → generation NLL, accuracy, P/R/F1, distinct-n, repetition, lengths
   │  write runs/<name>.json and generations/<name>.jsonl.gz
   ▼
run_beam_timing() → greedy_beam1_agreement() → aggregate() → summary.csv, metrics_by_run.csv → save_plots()
```

### 6.2 Setup and data

- Model `EleutherAI/pythia-160m` at revision `50f5173d…`, FP32, eval mode. TF32 matmuls are disabled so results are exact FP32.
- **`model.generate = None`** (`experiment.py:449`) guarantees no strategy can call Hugging Face's generator.
- ROCStories `prompt` = the story's first sentence; `continuation` = the other four sentences (the reference). `prepare_stories` keeps the original split index for each story, excludes empty references, and tokenises the reference with a **leading space** so it tokenises as it would after the prompt.
- References average 42.2 Pythia tokens (p99 61); `max_new_tokens = 64`.
- **Newline ban**: a story is one paragraph, and Pythia's top token after a sentence-final period is often a newline. Stopping at a newline would leave many greedy/beam outputs empty, so newline tokens are banned for every strategy (`newline_policy: ban`) and only EOS or the 64-token cap ends a continuation.
- Pythia's output layer has 50,304 logits (padded to a multiple of 128) but the tokenizer has 50,277 ids; the extra logits are masked to −inf.

### 6.3 `IncrementalLM` (`decoding.py:33`)

The wrapper owns everything the strategies should not have to think about:

- `left_pad(prompts)`: **left-pad** with EOS (Pythia has no pad token) and build an attention mask. With left padding, every row's next token is at the last position, so one `logits[:, -1]` serves the whole batch.
- `start(prompts)`: **position ids = `cumsum(mask) − 1`** (`decoding.py:81`), so padding does not shift the rotary positions of real tokens. One prefill forward with `use_cache=True`. Returns masked last-position logits and a `DecodeState` (cache, mask, next position, forward-call count).
- `step(tokens, state)`: append a 1 to the mask, feed one token per row with `position_ids = next_position` and the cache, increment positions.
- `_mask_logits`: cast to float32, set logits beyond the vocab and banned ids to −inf.
- `reorder(state, index)`: select or duplicate batch rows of the cache (`cache.reorder_cache`), mask and positions. Beam search uses it for expansion and re-ranking.

Contrast with Part 1, which right-pads and passes explicit positions with a masked padding gap. Both approaches keep RoPE positions correct.

### 6.4 Greedy and sampling (`_step_decode`, `decoding.py:149`)

One shared loop: `choose(logits)` picks one token per row, finished rows emit the pad id, `finished |= stop_mask[token]`, stop when all rows finish or at 64 tokens. `_truncate` cuts each row after its first stop token and records whether it stopped.

- **Greedy**: `choose = argmax`.
- **Sampling** (`sample_decode`): divide by temperature (1.0 here), then optional top-k, then optional top-p, then `softmax` and `torch.multinomial` with a `torch.Generator` seeded per run (seeds 0, 1, 2).
- **`top_k_filter`**: threshold = the k-th largest logit; everything below is −inf (ties at the threshold are kept).
- **`top_p_filter`**: sort descending, `mass_before = cumsum(p) − p` (the probability mass strictly before each token), remove tokens with `mass_before ≥ p`, scatter back to vocab order. This keeps the smallest set whose mass reaches p, **including** the token that crosses p, and the most likely token always survives (its `mass_before` is 0).
- **Pure sampling**: no truncation.

### 6.5 Beam search (`beam_search_decode`, `decoding.py:219`)

1. Prefill the B prompts once, then duplicate every row k times (`repeat_interleave`) and reorder the cache.
2. `beam_scores[:, 1:] = −inf` (`decoding.py:249`): all k copies are identical at first, so only one is allowed to expand. Otherwise the first step would produce k identical beams.
3. Each step: `log_softmax`, add each beam's cumulative score, flatten to `(B, k·V)`, and take the **top 2k** candidates. Up to k of them may be EOS, so 2k guarantees k live continuations remain.
4. Walk the candidates in rank order per example:
   - **EOS candidate**: if its rank is below k, store it as a finished hypothesis with normalised score `score / length^α` (α = 1; the length includes EOS). Either way it does not become a live beam.
   - Otherwise it fills the next live beam slot (score, token, source beam).
5. An example is **done** once it holds k finished hypotheses (Hugging Face's `early_stopping=True` semantics).
6. Reorder the cache by source beam and step with the new tokens.
7. At the end (or at 64 tokens), unfinished examples add their live beams as unfinished hypotheses. The winner maximises the normalised score.

With k = 1 this is exactly greedy decoding; a unit test confirms it. Another test brute-forces every sequence on a tiny vocabulary to show exhaustive beam search finds the true maximum-probability sequence.

### 6.6 Metrics (`metrics.py`)

| Metric | Definition | Code |
| --- | --- | --- |
| Generation perplexity | `exp(Σ NLL / Σ tokens)` of the strategy's own output (including EOS) given its prompt, under the **unmodified** model (T=1, no truncation, newlines not banned) | `continuation_nll`, `corpus_perplexity` |
| Reference perplexity | The same for the human continuation: **24.71** | `_reference_scores` |
| Accuracy | Position-wise token match against the reference's Pythia tokens, divided by the reference length; missing positions count as wrong | `positional_token_accuracy` |
| Precision / recall / F1 | SQuAD-style bag of words: lower-case, strip punctuation, multiset overlap; per story, then macro-averaged | `overlap_scores` |
| Distinct-1/2 | Corpus-level unique n-grams / all n-grams | `distinct_n` |
| Repeated-trigram fraction | Share of trigrams already seen earlier in the same text | `repeated_trigram_fraction` |
| Length, stopped, empty | Mean generated tokens, fraction ending in EOS, fraction empty | `_score_run` |

`continuation_nll` right-pads prompt + continuation, sets labels to −100 on prompt positions, and sums the per-token cross-entropy over continuation positions only. `mean_and_sem` gives the mean and standard error per run; `aggregate` reports the mean and standard deviation across seeds.

### 6.7 Resume and fingerprints

- `_fingerprint` = SHA-256 of the model, dataset and generation config plus every prompt's token ids.
- `_run_fingerprint` adds the strategy's exact settings, the seed, and the **source code bytes** of `decoding.py`, `metrics.py` and `experiment.py`. A finished run is reused only if this matches, so editing the decoder or changing `top_k` under the same name forces a rerun.
- `reference_perplexity.json` is cached by the data fingerprint.

### 6.8 Beam timing and complexity (`run_beam_timing`)

On a fixed 256-story subset, after a warm-up, every width k ∈ {1, 2, 4, 8, 16} is timed with 1 prompt per batch and with 32 prompts per batch (CUDA-synchronised; peak memory reset per run).

| k | 1/batch ms per forward | 1/batch ms per story | 32/batch rows | 32/batch ms per forward | 32/batch ms per story | Peak MiB (32/batch) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10.02 | 641 | 32 | 12.94 | 25.9 | 838 |
| 2 | 10.23 | 652 | 64 | 15.32 | 30.6 | 1049 |
| 4 | 10.38 | 657 | 128 | 19.82 | 39.6 | 1470 |
| 8 | 11.06 | 705 | 256 | 33.62 | 67.2 | 2313 |
| 16 | 10.90 | 696 | 512 | 64.67 | 129.3 | 3999 |

Complexity, with T generated tokens, vocabulary V, per-token network cost C, L layers and width d:

```math
\text{time} = O\big(T k C + k L d T^2 + T k V\big), \qquad \text{memory} = O\big(k L d (n_{\text{prompt}} + T) + k V\big)
```

Linear in k, with the same T sequential steps as greedy. On the GPU, one prompt at a time leaves the GPU idle, so latency per step is flat across widths. With 32 prompts, the cost per call approaches 2× per doubling once there are 256–512 rows. Memory is exactly linear (about 6.59 MiB per extra beam row). Batching matters far more than width: 32 prompts per batch cut the k=1 time per story from 641 ms to 25.9 ms.

### 6.9 Greedy vs beam-1 diagnostic

Greedy (batch 128) and beam-1 (batch 32) agree on **19,522 of 19,611** stories. `scripts/part3_greedy_beam1_diagnostic.py` reruns both on identical batches and measures the log-probability gap at each first divergence. In all 89 cases the two competing tokens are the model's top two, with a median gap of 3.7e-4: different batch widths change FP32 reduction order and flip near-ties. It is not a decoder bug. Beam-1 is also slower (513 s vs 132 s) only because its batch is 4× smaller, so it needs 4× more forward calls.

### 6.10 Results (19,611 stories; sampled rows are means over 3 seeds)

| Strategy | Accuracy | Precision | Recall | F1 | Gen PPL | Distinct-2 | Rep-3 | Len | Stop | Time s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | 0.0289 | 0.111 | 0.173 | 0.1333 | 2.19 | 0.028 | 0.714 | 63.9 | 0.003 | 132 |
| beam_1 | 0.0289 | 0.111 | 0.173 | 0.1334 | 2.19 | 0.028 | 0.714 | 63.9 | 0.003 | 513 |
| beam_2 | 0.0288 | 0.108 | 0.168 | 0.1299 | 2.05 | 0.028 | 0.723 | 63.8 | 0.013 | 609 |
| beam_4 | 0.0267 | 0.100 | 0.152 | 0.1187 | 1.90 | 0.029 | 0.742 | 63.4 | 0.043 | 791 |
| top_k 5 | 0.0208 | 0.161 | 0.248 | **0.1920** | 6.43 | 0.102 | 0.096 | 62.8 | 0.047 | 146 |
| top_k 20 | 0.0166 | 0.161 | 0.245 | 0.1911 | 11.18 | 0.160 | 0.020 | 62.0 | 0.065 | 146 |
| top_k 50 | 0.0152 | 0.157 | 0.236 | 0.1850 | 15.32 | 0.203 | 0.010 | 61.9 | 0.066 | 146 |
| top_p 0.5 | 0.0198 | 0.157 | 0.238 | 0.1860 | 6.77 | 0.126 | 0.104 | 62.7 | 0.043 | 176 |
| top_p 0.9 | 0.0137 | 0.147 | 0.214 | 0.1713 | **27.46** | 0.325 | 0.005 | 62.1 | 0.059 | 175 |
| top_p 0.95 | 0.0128 | 0.141 | 0.203 | 0.1639 | 38.92 | 0.381 | 0.003 | 62.1 | 0.060 | 175 |
| pure | 0.0119 | 0.134 | 0.189 | 0.1539 | 68.17 | 0.462 | 0.002 | 62.1 | 0.062 | 138 |

- **Deterministic decoding degenerates**: very low perplexity because it loops (71–74% repeated trigrams). A wider beam finds higher-likelihood text that is **more** repetitive and further from the reference: likelihood and quality diverge (Holtzman et al., 2020).
- **Truncated sampling gives the best overlap** (top-k 5 F1 0.192). Widening the candidate set raises diversity and perplexity and lowers F1 monotonically.
- **Top-p 0.9** comes closest to the human perplexity of 24.7.
- Accuracy is near the floor for every strategy because position-wise matching punishes any one-token shift; bag-of-words F1 is the more meaningful overlap.
- The model rarely emits EOS (0.3–6.6%), so most outputs run the full 64 tokens.
- Top-p 0.9 and pure sampling begin identically with seed 0: both draw the same random numbers.

---

## 7. Infrastructure: Ada, SLURM, publishing, report

**Ada layout.** Login via `ada-gw1`. Part 1 ran in `~/anlp_a2/part1_20260922`; Parts 2 and 3 in `~/anlp_a2/a2_20260928`. Every job sources `~/anlp_a2/slurm/env.sh`.

| Script | Purpose | Resources |
| --- | --- | --- |
| 07 notebook `PREPARE_SBATCH` | papermill `00_setup_and_checks` with `PART1_PREPARE=1` | 1 GPU, 20G, 1 h |
| 07 notebook `MODELS_SBATCH` | Array job 1–5, papermill each `model_N.ipynb` with `PART1_PROFILE=full` | 1 GPU each, 16G, 4 h |
| `part1_full_bleu_ada.sbatch` | papermill `10_full_test_bleu.ipynb` | 1 GPU, 30G, 12 h |
| `part2_ada.sbatch` | Synthetic check → sweep (if not done) → full runs | 1 GPU, 30G, 24 h |
| `part2_lion_check.sbatch <lr>` | Extra full-pass Lion run at one LR (sensitivity check only) | 1 GPU, 30G, 4 h |
| `part3_ada.sbatch [args]` | `main.py --part 3` with optional extra arguments | 1 GPU, 24G, 12 h |
| `part3_diagnostic_ada.sbatch` | Greedy vs beam-1 diagnostic on the same GPU type | 1 GPU, 16G, 2 h |
| `hf_upload_part2_ada.sbatch` | Upload Part 2 checkpoints from a compute node | CPU, 8G, 3 h |
| `ada_finalize.sh` | Hash Part 2 checkpoints, submit the diagnostic, upload to HF | login node |

All GPU jobs request `--partition=u22 --qos=medium --constraint=2080ti` and exclude the "phase3" 2080 Ti nodes (`gnode[065-066,068,070,073,075-081,083,087-090]`), whose CUDA initialisation failed on 2026-09-28.

**Publishing.**

- `scripts/upload_to_hf.py` is a **dry run unless `--execute`** is passed. Part 1 repos get the checkpoint, tokenizer, `implementation.ipynb`, full-test BLEU and predictions. Part 2 repos get the checkpoint, tokenizer, config, metrics, generations and `model.py`. SHA-256s are recorded in `assets/results/part2/v2/checkpoint_sha256.txt` and `hf_part2.json`.
- `scripts/upload_logs_to_wandb.py` rebuilds W&B runs from saved logs (Part 1 notebook records and checkpoint histories, Part 2 `progress_*.jsonl`). Nothing is recomputed; runs are tagged `uploaded-from-saved-logs`. Also a dry run without `--execute`.

**Report pipeline.**

1. Small result files are copied into `report/tables/`.
2. `report/make_report_tables.py` writes the LaTeX row bodies in `report/tex/*.tex`.
3. `report/make_report_figures.py` draws the compact PDF figures in `report/figures/report/`.
4. `report.tex` `\input`s those rows; build with `pdflatex report.tex` twice, then copy to `../report.pdf`.

No number in the report is typed by hand.

---

## 8. Tests

`uv run pytest` runs `src/part2/tests` and `src/part3/tests`. Part 1's checks live in its notebooks (Section 4.13).

| File | What it proves |
| --- | --- |
| `part2/tests/test_optimizer_equations.py` | AdamW, NAdamW, Lion and the Muon matrix path match independent scalar re-implementations over several steps, including clipping. NAdamW differs from AdamW. Newton–Schulz approximates the exact SVD polar factor. |
| `part2/tests/test_optimizers.py` | Every optimizer changes a parameter; decay semantics with zero and missing gradients; expected state tensors; Muon routing; Newton–Schulz is finite and shape-preserving; Sophia's clipped ratio matches a hand computation; Sophia rejects a non-finite Hessian estimate. |
| `part2/tests/test_overfit.py` | A tiny causal LM overfits with each optimizer. |
| `part2/tests/test_model.py` | Cached greedy generation equals full-forward generation. |
| `part2/tests/test_data.py` | Schema is inspected before fields are chosen; labels are shifted by one token. |
| `part2/tests/test_evaluate.py` | Continuation fits the context window at the boundary (a regression test). |
| `part2/tests/test_smoke.py` | Synthetic end-to-end run writes metrics and plots; sweep selects by validation loss; Sophia's Hessian refresh runs inside the loop; sweep extends an edge minimum and reuses finished trials. |
| `part3/tests/test_decoding.py` | Cached left-padded greedy equals uncached; cached step logits equal full forward; beam-1 equals greedy; top-k=1 and top-p→0 equal greedy; sampling is seed-reproducible and seed-dependent; exhaustive beam finds the best sequence; beam results do not depend on batching; top-k/top-p filters are correct; banned tokens never appear. |
| `part3/tests/test_metrics.py` | Bag-of-words overlap, positional accuracy with missing positions, diversity and perplexity helpers. |

---

## 9. Explaining the code: likely questions

**General**

1. *Why are the three parts separate projects?* Part 1 is notebook-driven (papermill on Ada) with its own uv environment pinning the CUDA 12.4 torch wheel. Parts 2 and 3 are a Python package run from `main.py`. The root `pyproject.toml` covers Parts 2–3; `src/part1/pyproject.toml` covers Part 1.
2. *How do you know runs are comparable?* Hashes: Part 1 asserts identical schedule digest, token counts, tokenizer hash, dataset revision and BLEU sample hash (`06_comparison`). Part 2 asserts every run starts from the same initial-state SHA-256. Part 3 fingerprints data, config and code.

**Part 1**

3. *Why decoder-only for translation?* The source and target are one sequence `[BOS] src [SEP] tgt [EOS]` with causal attention; the loss is masked to the target. It is the simplest architecture that lets the FFN be swapped in isolation.
4. *How does the model know the source language?* It does not get a language tag; the byte-level BPE keeps Vietnamese Latin script and Japanese kana/kanji distinct, so the language is implicit in the tokens.
5. *Why are expert widths 512/512/511/511?* The router's 2,048 parameters are taken out of the dense FFN's 2,097,152 budget, leaving 2,046 hidden units to split across four experts.
6. *Why not renormalise top-1 weights?* The weight would always be 1, so the router would get no gradient from the task loss.
7. *What does the balance loss do and why detach `f_e`?* It pushes routing towards uniform. `f_e` comes from a non-differentiable top-k, so the gradient goes through the mean router probability `P̄_e` instead.
8. *What is the z-loss for?* It penalises large router logits (`logsumexp²`) to keep the router softmax stable in FP16 training.
9. *How do you keep the backbone init identical across variants?* Every non-FFN matrix is re-initialised with a generator seeded from its parameter name, independent of how many FFN parameters were created first.
10. *Why does training abort on an AMP skip?* A skipped step would give that variant one fewer update; aborting guarantees every model takes exactly 12,473 updates.
11. *How does batched greedy decoding handle right padding?* New tokens are appended after the padding; the `valid` mask hides the gap and explicit positions `lens + t` keep RoPE correct.
12. *Why are MoE models slower despite fewer FLOPs?* Gathering tokens into four small expert matmuls with `torch.where`/`index_add_` is less GPU-efficient than one dense matmul at this scale.
13. *Do experts specialise by language?* No: by token role (source vs English) and depth. vi and ja routing distributions are nearly identical.

**Part 2**

14. *Why split by document family?* Human and AI versions of one document are near-duplicates; splitting rows randomly would leak test content into training.
15. *How is "one pass" defined?* 153,169 blocks × 256 = 39,211,264 prediction tokens, processed in a fixed permutation.
16. *Where is gradient clipping?* Inside each optimizer's `step()` (`common.global_gradient_scale`), on gradients that `GradScaler.step` has already unscaled.
17. *How do you detect skipped FP16 steps?* GradScaler lowers its scale exactly when it skips; the loop compares the scale before and after.
18. *AdamW vs NAdamW in one line?* NAdamW uses the look-ahead `β₁m + (1−β₁)g` in the numerator, applying momentum one step early.
19. *Why does Lion need a smaller LR and larger weight decay?* Its sign update moves every coordinate by ±η regardless of gradient size, so the effective step is larger than Adam's.
20. *What does Newton–Schulz do?* It approximately replaces the update matrix by its orthogonal polar factor (all singular values → 1), so no single direction dominates the step.
21. *Why do embeddings and LayerNorms not use Muon?* Muon is defined for hidden 2-D weight matrices; embeddings (a lookup table, here tied to the output head) and 1-D vectors use the AdamW fallback.
22. *What is the Gauss–Newton–Bartlett estimator?* Sample labels from the model's own predictive distribution and square the gradient of the loss on those labels; in expectation this is the Gauss–Newton diagonal.
23. *Why is Sophia close to Lion here?* About 94% of coordinates hit the `min(·, 1)` clip, so the update reduces to `η·sign(m)`.
24. *Why sweep on 0.1× of the data?* Cost. The catch: a compressed schedule rewards aggressive LRs, which is why Lion's selected LR was worse at full length.

**Part 3**

25. *How do you avoid `generate()`?* `model.generate = None`; every strategy uses `IncrementalLM`, which calls `model(...)` with `past_key_values`.
26. *Why left padding with explicit position ids?* So every row's next token is at the last column, and `cumsum(mask) − 1` keeps real-token positions starting at 0.
27. *Why 2k candidates in beam search?* Up to k candidates can be EOS; 2k guarantees k non-EOS live beams.
28. *Why set the extra beams to −inf at the start?* All k copies are identical; without it the first step would fill every beam with the same continuation.
29. *Why ban newlines?* Pythia often predicts a newline right after the prompt's period; stopping there would leave greedy/beam outputs empty.
30. *Why is the reference tokenised with a leading space?* So its tokens match what the model would produce right after the prompt (GPT-NeoX BPE marks word-initial spaces).
31. *Why is greedy perplexity lower than the human reference?* Greedy picks the most likely token every time and falls into loops, which the model finds very likely. Low perplexity here signals degeneration, not quality.
32. *Why do greedy and beam-1 differ on 89 stories?* Different batch sizes change padded width and FP32 reduction order; all 89 divergences are near-ties between the top two tokens.

---

## 10. Gotchas and loose ends

- **Stray untracked folder `part 1/`** at the repository root holds an older `model_1.ipynb`: the same code without outputs and without the final figure cell. Nothing references it. The real Part 1 code is in `src/part1/`.
- **The report's last line says code is in `part 1/`** (`report/report.tex:286`), but Part 1 lives in `src/part1/`.
- **Leakage filtering removed nothing.** The code drops overlapping targets and duplicate triplets, but `manifest.json` shows 0 removals. Only 54 triplets were dropped, all for length.
- **Two Part 1 BLEU numbers exist.** Model notebook headers show the 1,000-triplet sample (M1 35.13). The report table uses the full test split (M1 33.98) from `10_full_test_bleu.ipynb`.
- **Every Part 1 model chose the upper pilot LR (6e-4)**, a grid edge, so a larger LR might have helped.
- **Weight decay differs between parts.** Part 1 decays only matrices; Part 2 decays every parameter including LayerNorm, identically for all optimizers.
- **AMP skips are handled differently.** Part 1 aborts the run; Part 2 counts them (AdamW and NAdamW had 1 each).
- **Muon's fallback AdamW LR (6e-4) was not swept**, and it covers the large tied embedding.
- **Lion's extra runs** (`assets/results/part2/lion_lr_check/`) are outside the selection protocol and reported as a sensitivity check only.
- **Top-p 0.9 and pure sampling share random streams** with the same seed, so their outputs start identically.
- **`outputs/` and checkpoints are git-ignored.** Small copies of results live in `assets/results/`; checkpoints are on Hugging Face.

---

## 11. Function index

**Part 1** (`src/part1/shared/implementation.ipynb`; cell numbers are 0-based)

| Function / class | Cell | Role |
| --- | --- | --- |
| `Config`, `NAMES`, `seed_all`, `sha`, `atomic_torch`, `amp` | 1 | Hyperparameters and helpers |
| `MLP`, `FFN` | 3 | Dense MLP and the MoE layer |
| `Attention`, `Block`, `Transformer` | 3 | RoPE attention, pre-LN block, full model |
| `parameter_counts` | 3 | Total and active parameters on the meta device |
| `canonical`, `prepare_data` | 5 | Normalisation, audit, tokenizer, `.npy` splits |
| `PairStore`, `collate` | 5 | Memory-mapped splits, label masking |
| `epoch_batches`, `schedule`, `schedule_digest` | 5 | Length-bucketed batches, fixed budget, hash |
| `evaluate_nll` | 5 | Teacher-forced NLL/PPL per language |
| `training_run` | 6 | AdamW + FP16 training with resume |
| `greedy_translations`, `evaluate_bleu` | 6 | KV-cached greedy decoding, BLEU + bootstrap |
| `routing_diagnostics`, `plot_model_results` | 7 | Expert selection rates, entropy, heatmaps |
| `run_checks` | 7 | Correctness checks on a tiny model |

**Part 2**

| Function / class | Location |
| --- | --- |
| `run_part2` | `main.py:51` |
| `load_config`, `validate_config`, `with_overrides` | `src/part2/config.py:14`, `:25`, `:57` |
| `TokenBlockDataset` | `src/part2/data.py:26` |
| `choose_text_and_id_fields`, `deterministic_group_split` | `src/part2/data.py:135`, `:172` |
| `train_or_load_tokenizer`, `texts_to_blocks`, `prepare_real_data` | `src/part2/data.py:203`, `:250`, `:274` |
| `DenseTransformer`, `greedy_generate` | `src/part2/model.py:125`, `:179` |
| `learning_rate_scale`, `train_optimizer` | `src/part2/trainer.py:16`, `:97` |
| `_sampled_label_hessian_update` | `src/part2/trainer.py:74` |
| `evaluate_loss`, `evaluate_continuation_bleu` | `src/part2/evaluate.py:36`, `:65` |
| `make_optimizer`, `_prepare_data_and_initial_state` | `src/part2/experiment.py:57`, `:108` |
| `run_sweep`, `run_experiment` | `src/part2/experiment.py:174`, `:318` |
| `global_gradient_scale`, `decoupled_decay` | `src/part2/optimizers/common.py:29`, `:58` |
| `AdamW.step`, `NAdamW.step`, `Lion.step` | `adamw.py:45`, `nadamw.py:50`, `lion.py:48` |
| `newton_schulz_orthogonalize`, `is_muon_parameter`, `Muon.step` | `muon.py:20`, `:52`, `:176` |
| `SophiaG.update_hessian`, `SophiaG.step` | `sophia.py:75`, `:101` |

**Part 3**

| Function / class | Location |
| --- | --- |
| `run_part3` | `main.py:108` |
| `IncrementalLM` (`left_pad`, `start`, `step`, `reorder`) | `src/part3/decoding.py:33` (`:68`, `:79`, `:96`, `:113`) |
| `top_k_filter`, `top_p_filter` | `src/part3/decoding.py:120`, `:129` |
| `_step_decode`, `greedy_decode`, `sample_decode` | `src/part3/decoding.py:149`, `:176`, `:189` |
| `beam_search_decode` | `src/part3/decoding.py:219` |
| `overlap_scores`, `positional_token_accuracy` | `src/part3/metrics.py:33`, `:49` |
| `distinct_n`, `repeated_trigram_fraction` | `src/part3/metrics.py:60`, `:72` |
| `continuation_nll`, `corpus_perplexity` | `src/part3/metrics.py:88`, `:137` |
| `build_masks`, `prepare_stories` | `src/part3/experiment.py:71`, `:89` |
| `_decode_batches`, `_score_run` | `src/part3/experiment.py:125`, `:177` |
| `_run_fingerprint`, `run_beam_timing` | `src/part3/experiment.py:257`, `:309` |
| `aggregate`, `run_experiment` | `src/part3/experiment.py:389`, `:431` |
