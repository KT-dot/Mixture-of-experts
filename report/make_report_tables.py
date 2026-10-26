"""LaTeX table bodies for report.pdf, generated only from the files in report/tables/.

Run from the repository root:  .venv/bin/python report/make_report_tables.py
Writes report/tex/*.tex (table rows only; the tabular environments live in report.tex).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
OUT = ROOT / "tex"
OUT.mkdir(exist_ok=True)

SHORT_FFN = {
    1: "Dense",
    2: "4E top-1",
    3: "4E top-2",
    4: "Shared+3E top-1",
    5: "4E top-2 (act.)",
}


def write(name, rows):
    (OUT / name).write_text("\n".join(rows) + "\n")


def millions(n):
    return f"{n / 1e6:.2f}"


def part1_main():
    ppl = pd.read_csv(TABLES / "part1" / "part1_test_ppl_and_1000_triplet_bleu.csv").set_index("model")
    bleu = pd.read_csv(TABLES / "part1" / "part1_full_test_bleu.csv").set_index("model")
    rows = []
    for m in ppl.index:
        p, b = ppl.loc[m], bleu.loc[m]
        lo, hi = (float(v) for v in b["bleu_ci95"].split("|"))
        active = (
            millions(p["active_parameters_min"])
            if p["active_parameters_min"] == p["active_parameters_max"]
            else f"{p['active_parameters_min'] / 1e6:.3f}--{p['active_parameters_max'] / 1e6:.3f}"
        )
        rows.append(
            f"{m} & {SHORT_FFN[m]} & {millions(p['total_parameters'])} & {active} & "
            f"{p['test_ppl']:.3f} & {p['test_ppl_vi']:.3f} & {p['test_ppl_ja']:.3f} & "
            f"{b['bleu']:.2f} & {lo:.2f}--{hi:.2f} & {b['bleu_vi']:.2f} & {b['bleu_ja']:.2f} & "
            f"{b['bleu_unseen_sources']:.2f} & {p['training_minutes']:.1f} \\\\"
        )
    write("p1_main.tex", rows)


def part1_routing_summary():
    diag = pd.read_csv(TABLES / "part1" / "part1_expert_routing_diagnostics.csv")
    rows = []
    for m, sub in diag.groupby("model"):
        cells = []
        for region in ("source", "English-prefix"):
            r = sub[sub["region"] == region]
            first = r[r["layer"] == 1]
            cells += [
                f"{first['normalized_routing_entropy'].mean():.2f}",
                f"{r['normalized_routing_entropy'].mean():.2f}",
                f"{r['max_assignment_share'].max():.2f}",
            ]
        rows.append(f"{m} & " + " & ".join(cells) + f" & {int(sub['unused_experts'].sum())} \\\\")
    write("p1_routing.tex", rows)


def part2_main():
    metrics = pd.read_csv(TABLES / "part2" / "metrics.csv")
    names = {"adamw": "AdamW", "nadamw": "NAdamW", "lion": "Lion", "muon": "Muon", "sophia": "Sophia (bonus)"}
    sweep = pd.read_json(TABLES / "part2" / "lr_sweep_selected_hyperparameters.json")
    rows = []
    for opt in ["adamw", "nadamw", "lion", "muon", "sophia"]:
        sub = metrics[metrics["optimizer"] == opt].set_index("checkpoint_fraction")
        half, final = sub.loc[0.5], sub.loc[1.0]
        lr = sweep[opt]["selected_lr"]
        mult = sweep[opt]["selected_multiplier"]
        rows.append(
            f"{names[opt]} & {lr:g} ({mult:g}$\\times$) & {half['validation_loss']:.3f} & {final['validation_loss']:.3f} & "
            f"{final['validation_perplexity']:.1f} & {half['test_bleu']:.2f} & {final['test_bleu']:.2f} & "
            f"{final['training_wall_clock_seconds']:.0f} & {final['peak_gpu_memory_if_available'] / 2**30:.2f} & "
            f"{int(final['skipped_amp_steps'])} \\\\"
        )
    write("p2_main.tex", rows)


def part3_main():
    s = pd.read_csv(TABLES / "part3" / "summary.csv").sort_values("order")

    def cell(r, col, fmt):
        std = r.get(f"{col}_seed_std")
        if pd.notna(std):
            return f"{format(r[col], fmt)}$\\pm${format(std, fmt)}"
        return format(r[col], fmt)

    rows = []
    for _, r in s.iterrows():
        name = r["strategy"].replace("_", r"\_")
        rows.append(
            f"{name} & {cell(r, 'accuracy', '.4f')} & {r['precision']:.3f} & {r['recall']:.3f} & "
            f"{cell(r, 'f1', '.4f')} & {cell(r, 'generation_perplexity', '.2f')} & {r['distinct_2']:.3f} & "
            f"{r['repeated_trigram_fraction']:.3f} & {r['mean_generated_tokens']:.1f} & {r['stopped_fraction']:.3f} & "
            f"{r['decode_seconds']:.0f} \\\\"
        )
    write("p3_main.tex", rows)


def part3_timing():
    t = pd.read_csv(TABLES / "part3" / "beam_timing.csv")
    rows = []
    for k in sorted(t["beam_width"].unique()):
        a = t[(t["beam_width"] == k) & (t["batch_size"] == 1)].iloc[0]
        b = t[(t["beam_width"] == k) & (t["batch_size"] == 32)].iloc[0]
        rows.append(
            f"{k} & {a['ms_per_forward_call']:.2f} & {a['ms_per_story']:.0f} & {a['peak_gpu_memory_bytes'] / 2**20:.0f} & "
            f"{int(b['rows_per_forward'])} & {b['ms_per_forward_call']:.2f} & {b['ms_per_story']:.1f} & "
            f"{b['peak_gpu_memory_bytes'] / 2**20:.0f} \\\\"
        )
    write("p3_timing.tex", rows)


if __name__ == "__main__":
    part1_main()
    part1_routing_summary()
    part2_main()
    part3_main()
    part3_timing()
    for f in sorted(OUT.glob("*.tex")):
        print(f"== {f.name}")
        print(f.read_text(), end="")
