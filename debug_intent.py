"""
Diagnostic script for intent drift: run 1-2 tasks (e.g. failing ones), log intent vs instructions,
and print analysis questions. Use this to find why baseline beat intent on specific tasks.

Run from the intent_drift_experiment directory:
  python debug_intent.py
  python debug_intent.py --task summarization_0   # one task by type + index
  python debug_intent.py --tasks summarization_0,planning_1  # comma-separated

Set FAILING_TASK_IDS below to the task IDs where baseline beat intent (from task_comparison.csv winner=baseline).
"""
import argparse
import json
import sys
from pathlib import Path

# Add parent so "intent" and "metrics" resolve when run from repo root or intent_drift_experiment
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent import intent_factory
from metrics.ids import embedding, cosine_similarity, compute_ids, compute_goal_shift

# Fill in with task IDs where baseline beat intent (e.g. from task_comparison.csv, winner=baseline)
FAILING_TASK_IDS = ['summarization_1','summarization_3','planning_6', 'planning_4']  # e.g. ["summarization_0", "planning_4", "planning_5", "planning_7"]

TASKS_DIR = ROOT / "tasks"
# Match intent_agent: sim > 0.8 = refinement; sim < 0.6 = major_shift; else high_drift_risk
HIGH_THRESHOLD = 0.8
LOW_THRESHOLD = 0.6

def load_tasks():
    summarization = json.loads((TASKS_DIR / "summarization_tasks.json").read_text(encoding="utf-8"))
    planning = json.loads((TASKS_DIR / "planning_tasks.json").read_text(encoding="utf-8"))
    # Index by task_type + index for easy lookup (summarization_0, planning_1, ...)
    by_key = {}
    for i, t in enumerate(summarization):
        by_key["summarization_{}".format(i)] = ("summarization", t)
    for i, t in enumerate(planning):
        by_key["planning_{}".format(i)] = ("planning", t)
    return by_key


def debug_task(
    task_key: str,
    task_type: str,
    task: dict,
    run_agent: bool = True,
    model_name: str = "google/gemma-2-2b-it",
):
    """Run diagnostic for one task: intent replay (inferred per step), update_type, IDS per step, goal_shift task-level."""
    initial_intent = task["initial_intent"]
    steps = task.get("steps", [])
    initial_context = task.get("article") or task.get("context") or ""

    print("\n" + "=" * 60)
    print("DEBUG TASK: {} (id={})".format(task_key, task.get("id", "")))
    print("=" * 60)

    # 0. Context check
    has_context = bool(initial_context.strip())
    print("\n--- Context ---")
    print("  Has article/context: {} (len={})".format(has_context, len(initial_context)))
    if has_context:
        preview = initial_context[:120].replace("\n", " ") + ("..." if len(initial_context) > 120 else "")
        print("  Preview: \"{}\"".format(preview))

    # 1. Intent replay: inferred intent per step (same logic as interpretation)
    from metrics.intent_replay import intent_replay
    inferred = intent_replay(initial_intent, steps)
    print("\n--- Inferred intent per step (intent_replay) ---")
    print("  Thresholds: sim > {:.2f} = refinement (keep); sim < {:.2f} = major_shift; else high_drift_risk".format(
        HIGH_THRESHOLD, LOW_THRESHOLD))
    for r in inferred:
        goal_preview = (r["goal"][:60] + "...") if len(r["goal"]) > 60 else r["goal"]
        print("  Step {}: update_type={}  version={}  goal=\"{}\"".format(
            r["step"], r["update_type"], r["version"], goal_preview))
    goal_shift_task = compute_goal_shift(initial_intent, inferred[-1]["goal"]) if inferred else None
    if goal_shift_task is not None:
        print("  Goal shift (task-level cumulative): {:.4f}".format(goal_shift_task))

    # 2. Build intent and show prompt block
    intent = intent_factory(initial_intent, derive_from_goal=True)
    print("\n--- Initial Intent (to_prompt_block) ---")
    print(intent.to_prompt_block())

    # 3. Optionally run the agent and log responses
    if run_agent:
        print("\n--- Running IntentAgent (model={}) ---".format(model_name.split("/")[-1]))
        from agents.intent_agent import IntentAgent

        agent = IntentAgent(model_name=model_name, log_dir=str(ROOT / "logs"), seed=42)
        step_logs = agent.run_task(
            task_id=task_key + "_debug",
            initial_intent=initial_intent,
            steps=steps,
            initial_context=initial_context,
            verbose=True,
            seed=42,
        )
        inferred_by_step = {r["step"]: r for r in inferred}
        print("\n--- Step outputs: update_type, IDS (vs inferred goal), preview ---")
        asks_for_context_count = 0
        for log in step_logs:
            step = log.get("step", "?")
            prompt = log.get("prompt", "")[:60] + ("..." if len(log.get("prompt", "")) > 60 else "")
            update_type = log.get("update_type", "N/A")
            output = (log.get("output") or "")[:200]
            ids_val = None
            if step in inferred_by_step and log.get("output"):
                ids_val = compute_ids(log["output"], inferred_by_step[step]["goal"])
            ids_str = "{:.4f}".format(ids_val) if ids_val is not None else "N/A"
            asks_for = any(
                phrase in (log.get("output") or "").lower()
                for phrase in ["please provide", "provide the", "share the", "paste the", "provide more", "need the abstract", "await the", "when you have", "once provided", "once you provide", "full text of"]
            )
            if asks_for:
                asks_for_context_count += 1
            flag = " [ASKING FOR CONTEXT]" if asks_for else ""
            print("  Step {} (update_type={}, ids={}): prompt=\"{}\" -> output: \"{}...\"{}".format(
                step, update_type, ids_str, prompt, output, flag))
        if asks_for_context_count > 0:
            print("\n  [!] {} step(s) asked for context — check that article is retained in collapsed turns.".format(asks_for_context_count))
    else:
        print("\n--- Skipping agent run (use --run to execute) ---")
        step_logs = []

    # 4. Post-run analysis
    print("\n--- POST-RUN ANALYSIS ---")
    if run_agent and step_logs:
        high_risk = sum(1 for log in step_logs if log.get("update_type") == "high_drift_risk")
        major = sum(1 for log in step_logs if log.get("update_type") == "major_shift")
        if high_risk or major:
            print("  Update types: {} high_drift_risk, {} major_shift (goal replaced at those steps).".format(high_risk, major))
        # High-IDS steps for tips
        high_ids_steps = []
        if inferred_by_step and step_logs:
            for log in step_logs:
                step = log.get("step")
                if step in inferred_by_step and log.get("output"):
                    u = compute_ids(log["output"], inferred_by_step[step]["goal"])
                    if u >= 0.7:
                        high_ids_steps.append((step, u, (log.get("prompt") or "")[:50]))
        if high_ids_steps:
            print("  Steps with IDS >= 0.7: {} (review output vs goal for those steps).".format([s[0] for s in high_ids_steps]))
    print("  Key questions:")
    print("  1. Did inferred intent (replay) match expectations at each step?")
    print("  2. Was IDS high where the output drifted from the current goal?")
    print("  3. Did the model lose the article? (Look for 'please provide' / 'await the abstract'.)")
    print("\n--- WAYS TO REDUCE IDS ---")
    print("  • Prompt: Intent block now includes 'This turn: directly address and satisfy the current step'.")
    print("  • Metric: IDS uses semantic mode (max over sentences) and goal gist for long instructions.")
    print("  • Model: Try a larger model (e.g. Llama-3.1-8B) for better instruction following.")
    print("  • Task: Shorten or simplify instructions for high-USD steps; one clear ask per step.")
    print("  • Context: Ensure article/source is retained in the conversation for later steps.")
    print("  • Run: python debug_intent.py --task <key> --model meta-llama/Llama-3.1-8B-Instruct")
    print()


def main():
    parser = argparse.ArgumentParser(description="Debug intent on 1-2 tasks")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task keys, e.g. summarization_0,planning_1")
    parser.add_argument("--task", type=str, default=None, help="Single task key, e.g. summarization_0")
    parser.add_argument("--run", action="store_true", help="Run the IntentAgent (default: True if 1 task)")
    parser.add_argument("--no-run", action="store_true", help="Only print intent and similarities, do not run agent")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it",
                        help="Model to use for agent run (default: gemma-2-2b-it)")
    args = parser.parse_args()

    all_tasks = load_tasks()
    if args.tasks:
        keys = [k.strip() for k in args.tasks.split(",") if k.strip()]
    elif args.task:
        keys = [args.task.strip()]
    elif FAILING_TASK_IDS:
        keys = [k for k in FAILING_TASK_IDS if k in all_tasks][:2]
    else:
        # Default: first summarization and first planning
        keys = ["summarization_0", "planning_0"]

    run_agent = not args.no_run
    for key in keys:
        if key not in all_tasks:
            print("Unknown task key: {}. Available: {}".format(key, list(all_tasks.keys())[:10]))
            continue
        task_type, task = all_tasks[key]
        debug_task(
            key, task_type, task,
            run_agent=run_agent,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
