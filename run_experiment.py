"""
Repeatable experiment: same model, seed, temperature, tasks.
Run baseline agent then intent fusion agent; store logs separately.

How to interpret results:
- IDS (Intent Drift Score): 0 = aligned with initial intent, 1 = maximum drift. Lower is better.
- After a run, interpretation generates: CSV tables, PNG graphs, and experiment_report.md in logs/
- To re-run interpretation only (without re-running the experiment):
  python run_experiment.py --interpret-only
"""
import argparse
import csv
import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Fixed for repeatability (instructions)
SEED = 42
# Run each task 3 times with 3 different seeds for variance estimates.
SEEDS = [42, 43, 44]
N_RUNS = len(SEEDS)
TEMPERATURE = 0.7

# Paths relative to this file
ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
TASKS_DIR = ROOT / "tasks"
LOGS_DIR = ROOT / "logs"

# Models to run (RTX 3090 24GB). Override via --models model1,model2
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


def model_to_slug(model_name: str) -> str:
    """Filesystem-safe slug from model name (e.g. meta-llama/Llama-3.1-8B-Instruct -> llama-3.1-8b-instruct)."""
    # Take the part after the last "/" if present, else full name
    name = model_name.split("/")[-1] if "/" in model_name else model_name
    # Lowercase, replace spaces and special chars with underscore
    slug = "".join(c if c.isalnum() or c in "-." else "_" for c in name.lower())
    # Collapse multiple underscores and strip
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8").strip()


def load_tasks(filename: str) -> list:
    path = TASKS_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list:
    """Load JSONL file, return list of parsed objects."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    result = []
    for line in lines:
        if not line.strip():
            continue
        result.append(json.loads(line))
    return result


def _group_by_task_id(entries: list) -> dict:
    """Group log entries by task_id, sorted by step."""
    grouped = {}
    for e in entries:
        tid = e.get("task_id")
        if tid not in grouped:
            grouped[tid] = []
        grouped[tid].append(e)
    for tid in grouped:
        grouped[tid] = sorted(grouped[tid], key=lambda x: x.get("step", 0))
    return grouped


def _task_type(task_id: str) -> str:
    """Extract task type from task_id (e.g. summarization_0_run0 -> summarization)."""
    parts = task_id.split("_")
    if len(parts) >= 2:
        return parts[0]
    return "unknown"


def _task_metrics(steps: list) -> tuple:
    """Compute mean IDS, max IDS for steps 1..N (excluding step 0)."""
    ids_vals = []
    for s in steps:
        ids_val = s.get("ids")
        if ids_val is not None and s.get("step", 0) > 0:
            ids_vals.append(float(ids_val))
    if not ids_vals:
        return 0.0, 0.0
    return float(np.mean(ids_vals)), float(np.max(ids_vals))


def _paired_ids_significance(task_rows: list) -> dict:
    """
    Test whether the difference in mean IDS (baseline vs intent) across tasks is statistically significant.
    Uses paired data (same task under both conditions). Returns dict with p-values and effect size.
    """
    try:
        from scipy import stats
    except ImportError:
        return {"error": "scipy not installed", "n": len(task_rows)}

    baseline = np.array([r["baseline_mean_ids"] for r in task_rows])
    intent = np.array([r["intent_mean_ids"] for r in task_rows])
    n = len(baseline)
    if n < 2:
        return {"n": n, "p_ttest": None, "p_wilcoxon": None, "cohens_d": None}

    # Paired t-test: H0 = mean difference is 0
    t_stat, p_ttest = stats.ttest_rel(intent, baseline)

    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(intent, baseline, alternative="two-sided")
    except Exception:
        p_wilcoxon = None

    # Cohen's d for paired data: mean(D) / std(D), D = intent - baseline (negative = intent better)
    diff = intent - baseline
    std_diff = np.std(diff, ddof=1)
    cohens_d = (np.mean(diff) / std_diff) if std_diff > 0 else 0.0

    return {
        "n": n,
        "p_ttest": float(p_ttest),
        "p_wilcoxon": float(p_wilcoxon) if p_wilcoxon is not None else None,
        "cohens_d": float(cohens_d),
    }


def _ids_by_step(steps: list) -> list:
    """Return list of (step, ids) for all steps."""
    return [(s.get("step", 0), s.get("ids", 0.0)) for s in steps if s.get("ids") is not None]


def interpret_results(
    logs_dir: Path,
    baseline_path: Path = None,
    intent_path: Path = None,
    output_subdir: str = None,
) -> bool:
    """
    Load baseline and intent logs, compare using IDS, generate tables, graphs, and report.
    If baseline_path/intent_path are given, use those; else defaults to baseline_runs.jsonl and intent_runs.jsonl in logs_dir.
    If output_subdir is given, write all outputs (CSV, PNG, report) to logs_dir / output_subdir; else write to logs_dir.
    Returns True if interpretation succeeded, False if logs are missing/empty.
    """
    if baseline_path is None:
        baseline_path = logs_dir / "baseline_runs.jsonl"
    if intent_path is None:
        intent_path = logs_dir / "intent_runs.jsonl"
    out_dir = (logs_dir / output_subdir) if output_subdir else logs_dir
    if output_subdir:
        out_dir.mkdir(parents=True, exist_ok=True)

    baseline_entries = _load_jsonl(baseline_path)
    intent_entries = _load_jsonl(intent_path)

    if not baseline_entries or not intent_entries:
        print("  [Interpret] Skipping: missing or empty log files ({} / {}).".format(baseline_path, intent_path))
        return False

    baseline_by_task = _group_by_task_id(baseline_entries)
    intent_by_task = _group_by_task_id(intent_entries)
    all_task_ids = sorted(set(baseline_by_task.keys()) & set(intent_by_task.keys()))

    if not all_task_ids:
        print("  [Interpret] Skipping: no overlapping task_ids between logs.")
        return False

    # Per-task comparison
    task_rows = []
    for tid in all_task_ids:
        b_steps = baseline_by_task[tid]
        i_steps = intent_by_task[tid]
        b_mean, b_max = _task_metrics(b_steps)
        i_mean, i_max = _task_metrics(i_steps)
        delta_mean = i_mean - b_mean
        delta_max = i_max - b_max
        winner = "intent" if i_mean < b_mean else "baseline"
        task_rows.append({
            "task_id": tid,
            "task_type": _task_type(tid),
            "baseline_mean_ids": round(b_mean, 4),
            "intent_mean_ids": round(i_mean, 4),
            "baseline_max_ids": round(b_max, 4),
            "intent_max_ids": round(i_max, 4),
            "delta_mean": round(delta_mean, 4),
            "delta_max": round(delta_max, 4),
            "winner": winner,
        })

    # Aggregate stats
    intent_wins = sum(1 for r in task_rows if r["winner"] == "intent")
    total = len(task_rows)

    def agg_for_scope(rows):
        b_mean = np.mean([r["baseline_mean_ids"] for r in rows])
        i_mean = np.mean([r["intent_mean_ids"] for r in rows])
        wins = sum(1 for r in rows if r["winner"] == "intent")
        return b_mean, i_mean, wins, len(rows)

    overall_b, overall_i, _, _ = agg_for_scope(task_rows)
    sum_rows = [r for r in task_rows if r["task_type"] == "summarization"]
    plan_rows = [r for r in task_rows if r["task_type"] == "planning"]
    sum_b, sum_i, sum_wins, sum_n = agg_for_scope(sum_rows) if sum_rows else (0, 0, 0, 0)
    plan_b, plan_i, plan_wins, plan_n = agg_for_scope(plan_rows) if plan_rows else (0, 0, 0, 0)

    # --- Write CSV tables ---
    task_csv = out_dir / "task_comparison.csv"
    with open(task_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "task_id", "task_type", "baseline_mean_ids", "intent_mean_ids",
            "baseline_max_ids", "intent_max_ids", "delta_mean", "delta_max", "winner"
        ])
        w.writeheader()
        w.writerows(task_rows)

    # Statistical significance (paired: same task, baseline vs intent)
    sig = _paired_ids_significance(task_rows)

    summary_csv = out_dir / "summary_stats.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "baseline_mean_ids", "intent_mean_ids", "intent_wins", "total_tasks"])
        w.writerow(["overall", round(overall_b, 4), round(overall_i, 4), intent_wins, total])
        if sum_rows:
            w.writerow(["summarization", round(sum_b, 4), round(sum_i, 4), sum_wins, sum_n])
        if plan_rows:
            w.writerow(["planning", round(plan_b, 4), round(plan_i, 4), plan_wins, plan_n])

    # Write significance stats for programmatic use
    sig_csv = out_dir / "significance_stats.csv"
    with open(sig_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stat", "value"])
        w.writerow(["n_tasks", sig.get("n", "")])
        w.writerow(["p_ttest_paired", sig.get("p_ttest") if sig.get("p_ttest") is not None else ""])
        w.writerow(["p_wilcoxon_paired", sig.get("p_wilcoxon") if sig.get("p_wilcoxon") is not None else ""])
        cd = sig.get("cohens_d")
        w.writerow(["cohens_d_paired", round(cd, 4) if cd is not None else ""])

    # --- Generate graphs (matplotlib) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_step = 0
    for tid in all_task_ids:
        for s in baseline_by_task[tid]:
            max_step = max(max_step, s.get("step", 0))

    steps_range = list(range(max_step + 1))
    b_by_step = {s: [] for s in steps_range}
    i_by_step = {s: [] for s in steps_range}

    for tid in all_task_ids:
        for step, ids_val in _ids_by_step(baseline_by_task[tid]):
            if step in b_by_step:
                b_by_step[step].append(float(ids_val))
        for step, ids_val in _ids_by_step(intent_by_task[tid]):
            if step in i_by_step:
                i_by_step[step].append(float(ids_val))

    b_means = [np.mean(b_by_step[s]) if b_by_step[s] else 0 for s in steps_range]
    i_means = [np.mean(i_by_step[s]) if i_by_step[s] else 0 for s in steps_range]

    # ids_by_step.png
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_range, b_means, marker="o", label="Baseline", linewidth=2, markersize=6)
    ax.plot(steps_range, i_means, marker="s", label="Intent fusion", linewidth=2, markersize=6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean IDS")
    ax.set_title("Intent Drift Score by Step (mean across tasks)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ids_by_step.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ids_by_task_type.png
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    width = 0.35
    types_vals = []
    if sum_rows:
        types_vals.append(("summarization", sum_b, sum_i))
    if plan_rows:
        types_vals.append(("planning", plan_b, plan_i))
    if types_vals:
        labels = [t[0] for t in types_vals]
        b_vals = [t[1] for t in types_vals]
        i_vals = [t[2] for t in types_vals]
        x_pos = np.arange(len(labels))
        ax.bar(x_pos - width / 2, b_vals, width, label="Baseline")
        ax.bar(x_pos + width / 2, i_vals, width, label="Intent fusion")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean IDS")
        ax.set_title("Mean IDS by Task Type")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "ids_by_task_type.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ids_per_task.png (split by task type)
    type_rows = [(t, r) for t, r in [("summarization", sum_rows), ("planning", plan_rows)] if r]
    if type_rows:
        n_plots = len(type_rows)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=False)
        if n_plots == 1:
            axes = [axes]
        for idx, (task_type, rows) in enumerate(type_rows):
            ax = axes[idx]
            labels = [r["task_id"].replace("_run0", "") for r in rows]
            x_pos = np.arange(len(labels))
            width = 0.35
            ax.bar(x_pos - width / 2, [r["baseline_mean_ids"] for r in rows], width, label="Baseline")
            ax.bar(x_pos + width / 2, [r["intent_mean_ids"] for r in rows], width, label="Intent fusion")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Mean IDS")
            ax.set_title("Mean IDS per Task ({})".format(task_type))
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(out_dir / "ids_per_task.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Write Markdown report ---
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_lines = [
        "# Intent Drift Experiment Report",
        "",
        "Generated: {}".format(ts),
        "",
        "## Intent Drift Score (IDS)",
        "",
        "- **0** = aligned with initial intent",
        "- **1** = maximum semantic drift",
        "- Lower is better.",
        "",
        "## Summary",
        "",
        "| Scope | Baseline mean IDS | Intent mean IDS | Intent wins | Total tasks |",
        "|-------|-------------------|-----------------|-------------|-------------|",
        "| overall | {:.4f} | {:.4f} | {} | {} |".format(overall_b, overall_i, intent_wins, total),
    ]
    if sum_rows:
        report_lines.append("| summarization | {:.4f} | {:.4f} | {} | {} |".format(sum_b, sum_i, sum_wins, sum_n))
    if plan_rows:
        report_lines.append("| planning | {:.4f} | {:.4f} | {} | {} |".format(plan_b, plan_i, plan_wins, plan_n))

    report_lines.extend([
        "",
        "**Intent fusion had lower mean IDS in {}/{} tasks.**".format(intent_wins, total),
        "",
        "## Statistical significance",
        "",
    ])
    if "error" in sig:
        report_lines.append("- Could not compute (scipy not installed).")
    else:
        report_lines.append("- **Paired tests** (same task under baseline vs intent, n={} tasks).".format(sig["n"]))
        if sig.get("p_ttest") is not None:
            report_lines.append("- **Paired t-test** (H0: mean difference = 0): p = {:.4f}.".format(sig["p_ttest"]))
        if sig.get("p_wilcoxon") is not None:
            report_lines.append("- **Wilcoxon signed-rank** (non-parametric): p = {:.4f}.".format(sig["p_wilcoxon"]))
        if sig.get("cohens_d") is not None:
            report_lines.append("- **Cohen\'s d** (paired; negative = intent lower IDS): d = {:.3f}.".format(sig["cohens_d"]))
        report_lines.append("- Interpret: p < 0.05 suggests the mean IDS difference is unlikely due to chance; |d| ~ 0.2 small, ~0.5 medium, ~0.8+ large.")
    report_lines.extend([
        "",
        "## Task-Level Comparison",
        "",
        "| task_id | task_type | baseline_mean | intent_mean | delta | winner |",
        "|---------|-----------|---------------|-------------|-------|--------|",
    ])
    for r in task_rows:
        report_lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:+.4f} | {} |".format(
                r["task_id"], r["task_type"], r["baseline_mean_ids"], r["intent_mean_ids"],
                r["delta_mean"], r["winner"]
            )
        )
    report_lines.extend([
        "",
        "## Graphs",
        "",
        "![IDS by Step](ids_by_step.png)",
        "",
        "![IDS by Task Type](ids_by_task_type.png)",
        "",
        "![IDS per Task](ids_per_task.png)",
        "",
    ])

    (out_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # --- Console summary ---
    print("  Report: {}/experiment_report.md".format(out_dir))
    print("  Summary: Intent fusion had lower mean IDS in {}/{} tasks.".format(intent_wins, total))

    return True


def interpret_all_models(logs_dir: Path, models: list) -> bool:
    """
    Run interpretation for each model (per-model outputs in subdirs), then generate cross-model visualizations.
    Returns True if at least one model had valid logs.
    """
    slugs = [model_to_slug(m) for m in models]
    per_model_results = []  # list of (slug, overall_b, overall_i, intent_wins, total, steps_range, b_means, i_means)

    for slug in slugs:
        baseline_path = logs_dir / "baseline_runs_{}.jsonl".format(slug)
        intent_path = logs_dir / "intent_runs_{}.jsonl".format(slug)
        if not baseline_path.exists() or not intent_path.exists():
            print("  [Interpret] Skipping model {} (no logs).".format(slug))
            continue
        print("  [Interpret] Model: {} ...".format(slug))
        ok = interpret_results(logs_dir, baseline_path=baseline_path, intent_path=intent_path, output_subdir=slug)
        if ok:
            # Load summary for cross-model (we need overall_b, overall_i; and for ids_by_step_all_models we need per-step means)
            baseline_entries = _load_jsonl(baseline_path)
            intent_entries = _load_jsonl(intent_path)
            baseline_by_task = _group_by_task_id(baseline_entries)
            intent_by_task = _group_by_task_id(intent_entries)
            all_task_ids = sorted(set(baseline_by_task.keys()) & set(intent_by_task.keys()))
            if all_task_ids:
                task_rows = []
                for tid in all_task_ids:
                    b_mean, b_max = _task_metrics(baseline_by_task[tid])
                    i_mean, i_max = _task_metrics(intent_by_task[tid])
                    task_rows.append({"baseline_mean_ids": b_mean, "intent_mean_ids": i_mean})
                overall_b = float(np.mean([r["baseline_mean_ids"] for r in task_rows]))
                overall_i = float(np.mean([r["intent_mean_ids"] for r in task_rows]))
                intent_wins = sum(1 for r in task_rows if r["intent_mean_ids"] < r["baseline_mean_ids"])
                max_step = max(s.get("step", 0) for tid in all_task_ids for s in baseline_by_task[tid])
                steps_range = list(range(max_step + 1))
                b_by_step = {s: [] for s in steps_range}
                i_by_step = {s: [] for s in steps_range}
                for tid in all_task_ids:
                    for step, ids_val in _ids_by_step(baseline_by_task[tid]):
                        if step in b_by_step:
                            b_by_step[step].append(float(ids_val))
                    for step, ids_val in _ids_by_step(intent_by_task[tid]):
                        if step in i_by_step:
                            i_by_step[step].append(float(ids_val))
                b_means = [np.mean(b_by_step[s]) if b_by_step[s] else 0 for s in steps_range]
                i_means = [np.mean(i_by_step[s]) if i_by_step[s] else 0 for s in steps_range]
                per_model_results.append((slug, overall_b, overall_i, intent_wins, len(task_rows), steps_range, b_means, i_means))

    if not per_model_results:
        # Backward compatibility: try default log names (single-model run)
        if (logs_dir / "baseline_runs.jsonl").exists() and (logs_dir / "intent_runs.jsonl").exists():
            print("  [Interpret] Using default baseline_runs.jsonl / intent_runs.jsonl")
            return interpret_results(logs_dir)
        return False

    # Cross-model visualizations
    _write_cross_model_visualizations(logs_dir, per_model_results)
    return True


def _write_overall_experiment_report(logs_dir: Path, slugs: list):
    """
    Write logs/experiment_report.md aggregating all models, with overall p-value.
    Pools (model, task) pairs for a single paired test across the full experiment.
    """
    # Load task-level data from each model; deduplicate by base task (run0/1/2 are identical)
    pooled_rows = []
    for slug in slugs:
        tc_path = logs_dir / slug / "task_comparison.csv"
        if not tc_path.exists():
            continue
        with open(tc_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            seen = set()
            for row in reader:
                # Dedupe: one row per (slug, base_task) — drop _runN
                tid = row.get("task_id", "")
                base = re.sub(r"_run\d+$", "", tid) if tid else ""
                key = (slug, base)
                if key in seen:
                    continue
                seen.add(key)
                pooled_rows.append({
                    "baseline_mean_ids": float(row.get("baseline_mean_ids", 0)),
                    "intent_mean_ids": float(row.get("intent_mean_ids", 0)),
                    "task_type": row.get("task_type", ""),
                    "task_id": "{} ({})".format(base, slug),
                })

    if not pooled_rows:
        return

    # Overall stats from pooled data
    baseline_vals = [r["baseline_mean_ids"] for r in pooled_rows]
    intent_vals = [r["intent_mean_ids"] for r in pooled_rows]
    overall_b = float(np.mean(baseline_vals))
    overall_i = float(np.mean(intent_vals))
    intent_wins = sum(1 for r in pooled_rows if r["intent_mean_ids"] < r["baseline_mean_ids"])
    total = len(pooled_rows)

    # By task type
    sum_rows = [r for r in pooled_rows if r["task_type"] == "summarization"]
    plan_rows = [r for r in pooled_rows if r["task_type"] == "planning"]
    sum_b = float(np.mean([r["baseline_mean_ids"] for r in sum_rows])) if sum_rows else 0
    sum_i = float(np.mean([r["intent_mean_ids"] for r in sum_rows])) if sum_rows else 0
    sum_wins = sum(1 for r in sum_rows if r["intent_mean_ids"] < r["baseline_mean_ids"])
    plan_b = float(np.mean([r["baseline_mean_ids"] for r in plan_rows])) if plan_rows else 0
    plan_i = float(np.mean([r["intent_mean_ids"] for r in plan_rows])) if plan_rows else 0
    plan_wins = sum(1 for r in plan_rows if r["intent_mean_ids"] < r["baseline_mean_ids"])

    # Overall paired significance (pooled across models and tasks)
    sig = _paired_ids_significance(pooled_rows)

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    report_lines = [
        "# Intent Drift Experiment Report (Overall)",
        "",
        "Generated: {}".format(ts),
        "",
        "Aggregates all models: {}.".format(", ".join(slugs)),
        "",
        "## Intent Drift Score (IDS)",
        "",
        "- **0** = aligned with initial intent",
        "- **1** = maximum semantic drift",
        "- Lower is better.",
        "",
        "## Summary",
        "",
        "| Scope | Baseline mean IDS | Intent mean IDS | Intent wins | Total |",
        "|-------|-------------------|-----------------|-------------|-------|",
        "| overall | {:.4f} | {:.4f} | {} | {} |".format(overall_b, overall_i, intent_wins, total),
    ]
    if sum_rows:
        report_lines.append("| summarization | {:.4f} | {:.4f} | {} | {} |".format(sum_b, sum_i, sum_wins, len(sum_rows)))
    if plan_rows:
        report_lines.append("| planning | {:.4f} | {:.4f} | {} | {} |".format(plan_b, plan_i, plan_wins, len(plan_rows)))

    report_lines.extend([
        "",
        "**Intent fusion had lower mean IDS in {}/{} task-model pairs.**".format(intent_wins, total),
        "",
        "## Statistical Significance (Overall)",
        "",
    ])
    if "error" in sig:
        report_lines.append("- Could not compute (scipy not installed).")
    else:
        report_lines.append("- **Pooled paired tests** (all models × unique tasks, n={} pairs).".format(sig["n"]))
        if sig.get("p_ttest") is not None:
            report_lines.append("- **Paired t-test** (H0: mean difference = 0): p = {:.4e}.".format(sig["p_ttest"]))
        if sig.get("p_wilcoxon") is not None:
            report_lines.append("- **Wilcoxon signed-rank** (non-parametric): p = {:.4e}.".format(sig["p_wilcoxon"]))
        if sig.get("cohens_d") is not None:
            report_lines.append("- **Cohen's d** (paired; negative = intent lower IDS): d = {:.3f}.".format(sig["cohens_d"]))
        report_lines.append("- Interpret: p < 0.05 suggests the mean IDS difference is unlikely due to chance; |d| ~ 0.2 small, ~0.5 medium, ~0.8+ large.")

    report_lines.extend([
        "",
        "## Per-Model Reports",
        "",
    ])
    for slug in slugs:
        sub_report = slug + "/experiment_report.md"
        report_lines.append("- [{}]({})".format(slug, sub_report))
    report_lines.append("")
    report_lines.extend([
        "## Graphs",
        "",
        "![IDS by Model](ids_by_model.png)",
        "",
        "![IDS by Task Type by Model](ids_by_task_type_by_model.png)",
        "",
        "![IDS by Task Type (averaged)](ids_by_task_type_all_models.png)",
        "",
        "![IDS per Task (average across models)](ids_per_task_avg_models.png)",
        "",
        "![IDS by Step (all models)](ids_by_step_all_models.png)",
        "",
        "![IDS by Step (average across models)](ids_by_step_avg_models.png)",
        "",
    ])

    (logs_dir / "experiment_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print("  Overall report: logs/experiment_report.md (p = {})".format(
        "{:.4e}".format(sig["p_ttest"]) if sig.get("p_ttest") is not None else "N/A"
    ))


def _write_cross_model_visualizations(logs_dir: Path, per_model_results: list):
    """Write ids_by_model.png and ids_by_step_all_models.png to logs_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ids_by_model.png: bar chart of mean IDS (baseline vs intent) per model
    slugs = [r[0] for r in per_model_results]
    overall_b = [r[1] for r in per_model_results]
    overall_i = [r[2] for r in per_model_results]
    x = np.arange(len(slugs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(slugs) * 1.5), 5))
    ax.bar(x - width / 2, overall_b, width, label="Baseline")
    ax.bar(x + width / 2, overall_i, width, label="Intent fusion")
    ax.set_xticks(x)
    ax.set_xticklabels(slugs, rotation=45, ha="right")
    ax.set_ylabel("Mean IDS")
    ax.set_title("Mean IDS by Model")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(logs_dir / "ids_by_model.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ids_by_step_all_models.png: one line per model (intent fusion mean IDS by step)
    fig, ax = plt.subplots(figsize=(8, 5))
    for slug, _, _, _, _, steps_range, b_means, i_means in per_model_results:
        ax.plot(steps_range, i_means, marker="o", label="{} (intent)".format(slug), linewidth=1.5, markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean IDS (Intent fusion)")
    ax.set_title("Intent Drift Score by Step (all models, intent fusion)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(logs_dir / "ids_by_step_all_models.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ids_by_step_avg_models.png: two lines — baseline (avg across models) and intent (avg across models)
    max_step = max(s_range[-1] for _, _, _, _, _, s_range, _, _ in per_model_results)
    steps_common = list(range(max_step + 1))
    avg_b = []
    avg_i = []
    for step in steps_common:
        b_vals = [b_means[step] for _, _, _, _, _, s_range, b_means, i_means in per_model_results if step < len(s_range)]
        i_vals = [i_means[step] for _, _, _, _, _, s_range, b_means, i_means in per_model_results if step < len(s_range)]
        avg_b.append(np.mean(b_vals) if b_vals else np.nan)
        avg_i.append(np.mean(i_vals) if i_vals else np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps_common, avg_b, marker="o", label="Baseline (avg across models)", linewidth=2, markersize=6)
    ax.plot(steps_common, avg_i, marker="s", label="Intent fusion (avg across models)", linewidth=2, markersize=6)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean IDS")
    ax.set_title("Intent Drift Score by Step (average across all models)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(logs_dir / "ids_by_step_avg_models.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ids_by_task_type_by_model.png: summarization vs planning, baseline vs intent, models kept separate (no averaging)
    slugs = [r[0] for r in per_model_results]
    model_sum = {}  # slug -> (sum_b, sum_i)
    model_plan = {}  # slug -> (plan_b, plan_i)
    for slug in slugs:
        ss_path = logs_dir / slug / "summary_stats.csv"
        if not ss_path.exists():
            continue
        with open(ss_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scope = row.get("scope", "")
                if scope == "summarization":
                    model_sum[slug] = (float(row.get("baseline_mean_ids", 0)), float(row.get("intent_mean_ids", 0)))
                elif scope == "planning":
                    model_plan[slug] = (float(row.get("baseline_mean_ids", 0)), float(row.get("intent_mean_ids", 0)))
    if model_sum or model_plan:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        for idx, (task_type, data) in enumerate([("summarization", model_sum), ("planning", model_plan)]):
            if not data:
                continue
            ax = axes[idx]
            model_slugs = list(data.keys())
            x_pos = np.arange(len(model_slugs))
            width = 0.35
            b_vals = [data[s][0] for s in model_slugs]
            i_vals = [data[s][1] for s in model_slugs]
            ax.bar(x_pos - width / 2, b_vals, width, label="Baseline")
            ax.bar(x_pos + width / 2, i_vals, width, label="Intent fusion")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_slugs, rotation=45, ha="right")
            ax.set_ylabel("Mean IDS")
            ax.set_title("{}".format(task_type.capitalize()))
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
        if not model_sum:
            axes[0].axis("off")
        if not model_plan:
            axes[1].axis("off")
        fig.suptitle("Mean IDS by Task Type and Model (baseline vs intent)")
        fig.tight_layout()
        fig.savefig(logs_dir / "ids_by_task_type_by_model.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ids_by_task_type_all_models.png: mean IDS by task type, averaged across all models
    sum_b_vals, sum_i_vals, plan_b_vals, plan_i_vals = [], [], [], []
    for slug in slugs:
        ss_path = logs_dir / slug / "summary_stats.csv"
        if not ss_path.exists():
            continue
        with open(ss_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scope = row.get("scope", "")
                if scope == "summarization":
                    sum_b_vals.append(float(row.get("baseline_mean_ids", 0)))
                    sum_i_vals.append(float(row.get("intent_mean_ids", 0)))
                elif scope == "planning":
                    plan_b_vals.append(float(row.get("baseline_mean_ids", 0)))
                    plan_i_vals.append(float(row.get("intent_mean_ids", 0)))
    types_vals = []
    if sum_b_vals:
        types_vals.append(("summarization", np.mean(sum_b_vals), np.mean(sum_i_vals)))
    if plan_b_vals:
        types_vals.append(("planning", np.mean(plan_b_vals), np.mean(plan_i_vals)))
    if types_vals:
        labels = [t[0] for t in types_vals]
        b_vals = [t[1] for t in types_vals]
        i_vals = [t[2] for t in types_vals]
        x_pos = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(x_pos - width / 2, b_vals, width, label="Baseline")
        ax.bar(x_pos + width / 2, i_vals, width, label="Intent fusion")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean IDS")
        ax.set_title("Mean IDS by Task Type (average across all models)")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(logs_dir / "ids_by_task_type_all_models.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ids_per_task_avg_models.png: mean IDS per task (summarization_0, 1, 2, 3, planning_4, ...), averaged across all models
    # Collect per-task data from each model's task_comparison.csv; dedupe by base task; average across models
    task_to_b = {}  # base_task -> [baseline_mean_ids from each model]
    task_to_i = {}
    task_to_type = {}
    for slug in slugs:
        tc_path = logs_dir / slug / "task_comparison.csv"
        if not tc_path.exists():
            continue
        seen = set()
        with open(tc_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row.get("task_id", "")
                base = re.sub(r"_run\d+$", "", tid) if tid else ""
                if (slug, base) in seen:
                    continue
                seen.add((slug, base))
                if base not in task_to_b:
                    task_to_b[base] = []
                    task_to_i[base] = []
                    task_to_type[base] = row.get("task_type", "")
                task_to_b[base].append(float(row.get("baseline_mean_ids", 0)))
                task_to_i[base].append(float(row.get("intent_mean_ids", 0)))
    # Build rows: (base_task, task_type, mean_b, mean_i), grouped by task_type for subplots
    per_task_avg = []
    for base in sorted(task_to_b.keys(), key=lambda x: (task_to_type.get(x, ""), x)):
        per_task_avg.append({
            "task_id": base,
            "task_type": task_to_type.get(base, ""),
            "baseline_mean": np.mean(task_to_b[base]),
            "intent_mean": np.mean(task_to_i[base]),
        })
    if per_task_avg:
        sum_avg = [r for r in per_task_avg if r["task_type"] == "summarization"]
        plan_avg = [r for r in per_task_avg if r["task_type"] == "planning"]
        type_rows = [(t, r) for t, r in [("summarization", sum_avg), ("planning", plan_avg)] if r]
        n_plots = len(type_rows)
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=False)
        if n_plots == 1:
            axes = [axes]
        for idx, (task_type, rows) in enumerate(type_rows):
            ax = axes[idx]
            labels = [r["task_id"] for r in rows]
            x_pos = np.arange(len(labels))
            width = 0.35
            ax.bar(x_pos - width / 2, [r["baseline_mean"] for r in rows], width, label="Baseline")
            ax.bar(x_pos + width / 2, [r["intent_mean"] for r in rows], width, label="Intent fusion")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Mean IDS")
            ax.set_title("Mean IDS per Task ({}) — average across all models".format(task_type))
            ax.legend()
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(logs_dir / "ids_per_task_avg_models.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Optional: cross_model_summary.csv
    with open(logs_dir / "cross_model_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_slug", "baseline_mean_ids", "intent_mean_ids", "intent_wins", "total_tasks"])
        for r in per_model_results:
            w.writerow([r[0], round(r[1], 4), round(r[2], 4), r[3], r[4]])

    # Overall experiment report (aggregates all models, includes overall p-value)
    slugs = [r[0] for r in per_model_results]
    _write_overall_experiment_report(logs_dir, slugs)
    print("  Cross-model: ids_by_model.png, ids_by_task_type_by_model.png, ids_by_task_type_all_models.png, ids_per_task_avg_models.png, ids_by_step_all_models.png, ids_by_step_avg_models.png, cross_model_summary.csv")


def main(models_override: list = None):
    models_to_run = models_override if models_override is not None else MODELS
    print("=" * 60)
    print("Intent Drift Experiment")
    print("=" * 60)

    print("\n[1/5] Models: {} ({} total).".format([model.split("/")[-1] for model in models_to_run], len(models_to_run)))
    print("      Seeds: {} ({} runs per task).".format(SEEDS, N_RUNS))

    print("[2/5] Preparing logs directory: {}".format(LOGS_DIR))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def model_has_logs(slug):
        return (LOGS_DIR / "baseline_runs_{}.jsonl".format(slug)).exists() and (
            LOGS_DIR / "intent_runs_{}.jsonl".format(slug)
        ).exists()

    for model in models_to_run:
        slug = model_to_slug(model)
        if model_has_logs(slug):
            continue
        for name in ("baseline_runs_{}.jsonl", "intent_runs_{}.jsonl"):
            p = LOGS_DIR / name.format(slug)
            if p.exists():
                p.write_text("")
    print("      Cleared previous run logs for models that will be run (kept existing logs for others).")

    print("[3/5] Loading prompts and tasks...")
    system_prompt_baseline = load_prompt("agent_base.txt")
    summarization_tasks = load_tasks("summarization_tasks.json")
    planning_tasks = load_tasks("planning_tasks.json")
    all_tasks = [
        ("summarization", summarization_tasks),
        ("planning", planning_tasks),
    ]
    total_tasks = sum(len(tasks) for _, tasks in all_tasks)
    total_runs = total_tasks * N_RUNS * len(models_to_run)
    print("      Loaded baseline prompt, {} task sets.".format(total_tasks))

    # Build flat list of (task_type, task_id_global, task, run_idx) for progress bar
    def iter_baseline_runs():
        task_id_global = 0
        for task_type, task_list in all_tasks:
            for task in task_list:
                for run in range(N_RUNS):
                    yield task_type, task_id_global, task, run
                task_id_global += 1

    runs_list = list(iter_baseline_runs())

    for model in models_to_run:
        slug = model_to_slug(model)
        if model_has_logs(slug):
            print("\n[4/5] Model: {} (slug: {}) — skipping (log files already exist).".format(model.split("/")[-1], slug))
            continue
        baseline_log = "baseline_runs_{}.jsonl".format(slug)
        intent_log = "intent_runs_{}.jsonl".format(slug)
        print("\n[4/5] Model: {} (slug: {})".format(model.split("/")[-1], slug))
        from agents.baseline_agent import BaselineAgent
        from agents.intent_agent import IntentAgent

        baseline_agent = BaselineAgent(log_dir=str(LOGS_DIR), model_name=model, seed=SEED)
        intent_agent = IntentAgent(log_dir=str(LOGS_DIR), model_name=model, seed=SEED)

        print("--- Baseline agent (no intent fusion) ---")
        for task_type, task_id_global, task, run in tqdm(runs_list, desc="Baseline runs", unit="run"):
            task_id = "{}_{}_run{}".format(task_type, task_id_global, run)
            run_seed = SEEDS[run]
            set_global_seed(run_seed)
            initial_context = task.get("article") or task.get("context") or ""
            baseline_agent.run_and_log(
                task_id=task_id,
                initial_intent=task["initial_intent"],
                steps=task["steps"],
                system_prompt=system_prompt_baseline,
                initial_context=initial_context,
                log_file=baseline_log,
                verbose=False,
                seed=run_seed,
            )

        print("\n--- Freeing baseline model from GPU ---")
        baseline_agent._model = None
        baseline_agent._tokenizer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n--- Intent fusion agent (Intent object sent every time) ---")
        for task_type, task_id_global, task, run in tqdm(runs_list, desc="Intent fusion runs", unit="run"):
            task_id = "{}_{}_run{}".format(task_type, task_id_global, run)
            run_seed = SEEDS[run]
            set_global_seed(run_seed)
            initial_context = task.get("article") or task.get("context") or ""
            intent_agent.run_and_log(
                task_id=task_id,
                initial_intent=task["initial_intent"],
                steps=task["steps"],
                initial_context=initial_context,
                log_file=intent_log,
                verbose=False,
                seed=run_seed,
            )

        # Free intent model before loading next model
        intent_agent._model = None
        intent_agent._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[5/5] Interpretation (per model and cross-model)...")
    interpret_all_models(LOGS_DIR, models_to_run)
    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("  Logs: {}".format(LOGS_DIR))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intent Drift Experiment")
    parser.add_argument("--interpret-only", action="store_true",
                        help="Skip experiment; only regenerate interpretation from existing logs")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names to run (default: built-in MODELS list)")
    args = parser.parse_args()

    models_override = None
    if args.models:
        models_override = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.interpret_only:
        print("=" * 60)
        print("Interpret results only (from existing logs)")
        print("=" * 60)
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        models_for_interpret = models_override if models_override is not None else MODELS
        if interpret_all_models(LOGS_DIR, models_for_interpret):
            print("\nInterpretation complete.")
        else:
            print("\nNo logs found. Run the full experiment first.")
        print("=" * 60)
    else:
        main(models_override=models_override)
