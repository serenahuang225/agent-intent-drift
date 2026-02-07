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
from metrics.ids import embedding, cosine_similarity

# Fill in with task IDs where baseline beat intent (e.g. from task_comparison.csv, winner=baseline)
FAILING_TASK_IDS = []  # e.g. ["summarization_0", "planning_4", "planning_5", "planning_7"]

TASKS_DIR = ROOT / "tasks"
INTENT_SIM_THRESHOLD = 0.30  # match intent_agent.py LOW_THRESHOLD; below this = conflict

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
    sim_threshold: float = INTENT_SIM_THRESHOLD,
):
    """Run diagnostic for one task: print intent block, step-by-step similarity, and optionally run agent."""
    initial_intent = task["initial_intent"]
    steps = task.get("steps", [])
    initial_context = task.get("article") or task.get("context") or ""

    print("\n" + "=" * 60)
    print("DEBUG TASK: {} (id={})".format(task_key, task.get("id", "")))
    print("=" * 60)

    # 0. Context check (critical for summarization — if missing, model will ask for it)
    has_context = bool(initial_context.strip())
    print("\n--- Context ---")
    print("  Has article/context: {} (len={})".format(has_context, len(initial_context)))
    if has_context:
        preview = initial_context[:120].replace("\n", " ") + ("..." if len(initial_context) > 120 else "")
        print("  Preview: \"{}\"".format(preview))

    # 1. Build intent and show prompt block
    intent = intent_factory(initial_intent, derive_from_goal=True)
    print("\n--- Initial Intent (to_prompt_block) ---")
    print(intent.to_prompt_block())

    # 2. For each step, show instruction vs goal similarity
    print("\n--- Step-by-step: Instruction vs Intent Goal (similarity) ---")
    print("  Threshold: sim < {:.2f} → conflict".format(sim_threshold))
    goal_emb = embedding(intent.goal)
    for i, instruction in enumerate(steps):
        inst_emb = embedding(instruction)
        sim = cosine_similarity(goal_emb, inst_emb)
        conflict = sim < sim_threshold
        print("  Step {}: sim={:.3f}  conflict={}  instruction: \"{}\"".format(
            i, sim, conflict, instruction[:70] + ("..." if len(instruction) > 70 else "")))

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
        print("\n--- Step outputs (preview) ---")
        asks_for_context_count = 0
        for log in step_logs:
            step = log.get("step", "?")
            prompt = log.get("prompt", "")[:60] + ("..." if len(log.get("prompt", "")) > 60 else "")
            conflict = log.get("conflict_would_trigger", "N/A")
            output = (log.get("output") or "")[:200]
            # Detect "asking for context" (article dropped or never provided)
            asks_for = any(
                phrase in (log.get("output") or "").lower()
                for phrase in ["please provide", "provide the", "share the", "paste the", "provide more", "need the abstract", "await the", "when you have", "once provided", "once you provide", "full text of"]
            )
            if asks_for:
                asks_for_context_count += 1
            flag = " [ASKING FOR CONTEXT]" if asks_for else ""
            print("  Step {} (conflict={}): prompt=\"{}\" -> output: \"{}...\"{}".format(
                step, conflict, prompt, output, flag))
        if asks_for_context_count > 0:
            print("\n  [!] {} step(s) asked for context — check that article is retained in collapsed turns.".format(asks_for_context_count))
    else:
        print("\n--- Skipping agent run (use --run to execute) ---")

    # 4. Post-run analysis
    print("\n--- POST-RUN ANALYSIS ---")
    if run_agent and step_logs:
        # Auto-analyze
        conflict_count = sum(1 for log in step_logs if log.get("conflict_would_trigger") is True)
        if conflict_count == len(steps) - 1 and len(steps) > 1:
            print("  [!] All step instructions flagged as conflict (sim < 0.5).")
            print("      Sub-tasks often have low similarity to the parent goal — consider a hierarchical")
            print("      similarity check or raising the threshold for elaborative chains.")
    print("  Key questions:")
    print("  1. Was the intent's constraint clear for the conflicting instruction?")
    print("  2. Did the instruction directly contradict the goal (e.g., 'add detail' vs. 'be concise')?")
    print("  3. Did the agent's output violate a success criterion (e.g., was it >1 sentence)?")
    print("  4. Did the model lose the article? (Look for 'please provide' / 'await the abstract'.)")
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
    parser.add_argument("--threshold", type=float, default=INTENT_SIM_THRESHOLD,
                        help="Similarity threshold for conflict (sim < N → conflict, default=0.5)")
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
            sim_threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
