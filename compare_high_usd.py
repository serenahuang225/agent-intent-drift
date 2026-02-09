"""
Visualizer: compare steps where baseline had high IDS but intent model had lower IDS.

Finds steps where the baseline drifted a lot (high Intent Drift Score) but the intent agent stayed
on goal (lower IDS), and writes an HTML report with side-by-side prompt, inferred goal,
baseline output, and intent output.

Usage (from intent_drift_experiment directory):
  python compare_high_usd.py --model llama-3.1-8b-instruct
  python compare_high_usd.py --model gemma-2-2b-it --min-baseline-ids 0.75 --min-gap 0.2
  python compare_high_usd.py --model mistral-7b-instruct-v0.2 --output logs/intent_wins.html
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
TASKS_DIR = ROOT / "tasks"


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        result.append(json.loads(line))
    return result


def _group_by_task_id(entries: list) -> dict:
    grouped = {}
    for e in entries:
        tid = e.get("task_id")
        if tid not in grouped:
            grouped[tid] = []
        grouped[tid].append(e)
    for tid in grouped:
        grouped[tid] = sorted(grouped[tid], key=lambda x: x.get("step", 0))
    return grouped


def _get_task_def(task_id: str, tasks_dir: Path) -> tuple:
    base = re.sub(r"_run\d+$", "", task_id)
    parts = base.split("_")
    if len(parts) < 2:
        return None, None
    task_type, idx_str = parts[0], parts[1]
    try:
        global_idx = int(idx_str)
    except ValueError:
        return None, None
    sum_path = tasks_dir / "summarization_tasks.json"
    plan_path = tasks_dir / "planning_tasks.json"
    n_sum = len(json.loads(sum_path.read_text(encoding="utf-8"))) if sum_path.exists() else 0
    n_plan = len(json.loads(plan_path.read_text(encoding="utf-8"))) if plan_path.exists() else 0
    if task_type == "summarization":
        local_idx = global_idx
        path = sum_path
    elif task_type == "planning":
        local_idx = global_idx - n_sum
        path = plan_path
    else:
        path = tasks_dir / "{}_tasks.json".format(task_type)
        local_idx = global_idx
    if not path.exists() or local_idx < 0:
        return None, None
    task_list = json.loads(path.read_text(encoding="utf-8"))
    if local_idx >= len(task_list):
        return None, None
    task = task_list[local_idx]
    return task.get("initial_intent"), task.get("steps")


def _slug_from_model_arg(model_arg: str) -> str:
    """If user passed full name like meta-llama/Llama-3.1-8B-Instruct, convert to slug."""
    if "/" in model_arg:
        name = model_arg.split("/")[-1]
        slug = "".join(c if c.isalnum() or c in "-." else "_" for c in name.lower())
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_")
    return model_arg


def find_high_baseline_low_intent_ids(
    logs_dir: Path,
    slug: str,
    tasks_dir: Path,
    min_baseline_ids: float = 0.7,
    min_gap: float = 0.15,
):
    """Load logs, enrich with IDS via intent_replay, return list of (task_id, step, prompt, goal, b_out, i_out, b_ids, i_ids)."""
    from metrics.ids import compute_ids
    from metrics.intent_replay import intent_replay

    baseline_path = logs_dir / "baseline_runs_{}.jsonl".format(slug)
    intent_path = logs_dir / "intent_runs_{}.jsonl".format(slug)
    if not baseline_path.exists() or not intent_path.exists():
        return None, None, []

    baseline_entries = _load_jsonl(baseline_path)
    intent_entries = _load_jsonl(intent_path)
    baseline_by_task = _group_by_task_id(baseline_entries)
    intent_by_task = _group_by_task_id(intent_entries)
    all_task_ids = sorted(set(baseline_by_task.keys()) & set(intent_by_task.keys()))

    # Enrich with IDS (Intent Drift Score)
    for tid in all_task_ids:
        initial_intent, steps = _get_task_def(tid, tasks_dir)
        if initial_intent is None or steps is None:
            continue
        inferred = intent_replay(initial_intent, steps)
        if not inferred:
            continue
        inferred_by_step = {r["step"]: r for r in inferred}
        for s in baseline_by_task[tid]:
            step_i = s.get("step", 0)
            if step_i in inferred_by_step and "output" in s:
                s["ids"] = round(compute_ids(s["output"], inferred_by_step[step_i]["goal"]), 4)
        for s in intent_by_task[tid]:
            step_i = s.get("step", 0)
            if step_i in inferred_by_step and "output" in s:
                s["ids"] = round(compute_ids(s["output"], inferred_by_step[step_i]["goal"]), 4)

    # Find steps where baseline IDS high and intent IDS lower (intent helped)
    results = []
    for tid in all_task_ids:
        b_steps = {s["step"]: s for s in baseline_by_task[tid]}
        i_steps = {s["step"]: s for s in intent_by_task[tid]}
        initial_intent, steps = _get_task_def(tid, tasks_dir)
        if initial_intent is None or steps is None:
            continue
        inferred = intent_replay(initial_intent, steps)
        inferred_by_step = {r["step"]: r for r in inferred}
        for step in b_steps:
            if step not in i_steps or step not in inferred_by_step:
                continue
            b_s, i_s = b_steps[step], i_steps[step]
            b_ids = b_s.get("ids") or b_s.get("usd")
            i_ids = i_s.get("ids") or i_s.get("usd")
            if b_ids is None or i_ids is None:
                continue
            if b_ids >= min_baseline_ids and (b_ids - i_ids) >= min_gap:
                results.append({
                    "task_id": tid,
                    "step": step,
                    "prompt": b_s.get("prompt", ""),
                    "goal": inferred_by_step[step]["goal"],
                    "baseline_output": b_s.get("output", ""),
                    "intent_output": i_s.get("output", ""),
                    "baseline_ids": b_ids,
                    "intent_ids": i_ids,
                    "gap": round(b_ids - i_ids, 4),
                })
    return baseline_by_task, intent_by_task, results


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_html_report(
    results: list,
    slug: str,
    output_path: Path,
    max_output_chars: int = 1200,
):
    """Write an HTML file with side-by-side comparison for each result row. No .format() to avoid brace/placeholder issues."""
    rows_html = []
    for r in results:
        prompt = escape_html((r["prompt"] or "")[:500])
        goal = escape_html((r["goal"] or "")[:500])
        b_out = escape_html((r["baseline_output"] or "")[:max_output_chars])
        if len((r["baseline_output"] or "")) > max_output_chars:
            b_out += " …"
        i_out = escape_html((r["intent_output"] or "")[:max_output_chars])
        if len((r["intent_output"] or "")) > max_output_chars:
            i_out += " …"
        task_id_esc = escape_html(r["task_id"])
        b_ids = r.get("baseline_ids", r.get("baseline_usd", 0))
        i_ids = r.get("intent_ids", r.get("intent_usd", 0))
        gap = r["gap"]
        step = r["step"]
        row = (
            '<div class="card">'
            '<div class="meta">' + task_id_esc + ' · Step ' + str(step) + ' &nbsp;|&nbsp; Baseline IDS: <strong>' + str(b_ids) + '</strong> &rarr; Intent IDS: <strong>' + str(i_ids) + '</strong> (gap: ' + str(gap) + ')</div>'
            '<div class="prompt"><strong>Instruction:</strong> ' + prompt + '</div>'
            '<div class="goal"><strong>Inferred goal:</strong> ' + goal + '</div>'
            '<div class="side-by-side">'
            '<div class="panel baseline">'
            '<div class="badge high">Baseline (IDS ' + str(b_ids) + ')</div>'
            '<pre class="output">' + b_out + '</pre>'
            '</div>'
            '<div class="panel intent">'
            '<div class="badge low">Intent (IDS ' + str(i_ids) + ')</div>'
            '<pre class="output">' + i_out + '</pre>'
            '</div>'
            '</div>'
            '</div>'
        )
        rows_html.append(row)

    slug_esc = escape_html(slug)
    n = len(results)
    rows_joined = "\n".join(rows_html)
    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>High baseline IDS vs intent – " + slug_esc + "</title>\n"
        "<style>\n"
        "  body { font-family: system-ui, sans-serif; margin: 1rem 2rem; background: #f5f5f5; }\n"
        "  h1 { color: #333; }\n"
        "  .subtitle { color: #666; margin-bottom: 1.5rem; }\n"
        "  .card { background: #fff; border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,.1); }\n"
        "  .meta { color: #555; font-size: 0.9rem; margin-bottom: 0.5rem; }\n"
        "  .prompt, .goal { margin: 0.5rem 0; font-size: 0.95rem; color: #333; }\n"
        "  .goal { color: #555; }\n"
        "  .side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 0.75rem; }\n"
        "  @media (max-width: 900px) { .side-by-side { grid-template-columns: 1fr; } }\n"
        "  .panel { border-radius: 6px; padding: 0.75rem; }\n"
        "  .panel.baseline { background: #fff5f5; border: 1px solid #fcc; }\n"
        "  .panel.intent { background: #f5fff5; border: 1px solid #cfc; }\n"
        "  .badge { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem; }\n"
        "  .badge.high { color: #c00; }\n"
        "  .badge.low { color: #080; }\n"
        "  pre.output { white-space: pre-wrap; word-break: break-word; margin: 0; font-size: 0.85rem; line-height: 1.4; }\n"
        "</style>\n</head>\n<body>\n"
        "<h1>Intent vs baseline: high-IDS steps where intent did better</h1>\n"
        "<p class=\"subtitle\">Model: <strong>" + slug_esc + "</strong> &nbsp;|&nbsp; " + str(n) + " step(s) with high baseline IDS and lower intent IDS</p>\n"
        + rows_joined + "\n"
        "</body>\n</html>"
    )
    output_path.write_text(html, encoding="utf-8")
    print("Wrote: " + str(output_path))


def main():
    parser = argparse.ArgumentParser(
        description="Compare steps where baseline had high IDS but intent model had lower IDS; output HTML report.",
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Model slug (e.g. llama-3.1-8b-instruct) or full name (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--logs-dir", type=str, default=None,
                        help="Logs directory (default: intent_drift_experiment/logs)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output HTML path (default: logs/<slug>_intent_wins.html)")
    parser.add_argument("--min-baseline-ids", type=float, default=0.7,
                        help="Minimum baseline IDS to include a step (default: 0.7)")
    parser.add_argument("--min-gap", type=float, default=0.15,
                        help="Minimum (baseline_ids - intent_ids) to include (default: 0.15)")
    args = parser.parse_args()

    slug = _slug_from_model_arg(args.model)
    logs_dir = Path(args.logs_dir) if args.logs_dir else LOGS_DIR
    tasks_dir = TASKS_DIR
    output_path = Path(args.output) if args.output else (logs_dir / "{}_intent_wins.html".format(slug))

    baseline_by_task, intent_by_task, results = find_high_baseline_low_intent_ids(
        logs_dir, slug, tasks_dir,
        min_baseline_ids=args.min_baseline_ids,
        min_gap=args.min_gap,
    )
    if baseline_by_task is None:
        print("Error: could not load logs for model '{}' (slug: {}). Check that baseline_runs_{}.jsonl and intent_runs_{}.jsonl exist in {}.".format(
            args.model, slug, slug, slug, logs_dir,
        ))
        return 1
    if not results:
        print("No steps found with baseline IDS >= {} and gap >= {} for model {}.".format(
            args.min_baseline_ids, args.min_gap, slug,
        ))
        print("Try lowering --min-baseline-ids or --min-gap.")
        return 0
    print("Found {} step(s) where baseline had high IDS but intent had lower IDS (model: {}).".format(len(results), slug))
    write_html_report(results, slug, output_path)
    return 0


if __name__ == "__main__":
    exit(main())
