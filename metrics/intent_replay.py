"""
Intent replay: reapply the same intent-update logic (thresholds, with_goal_replacement)
to an instruction sequence without running the LLM. Used to get inferred intent per step
for IDS and goal_shift computation during interpretation.
"""
from typing import Any, Dict, List

from metrics.ids import cosine_similarity, embedding

# Same thresholds as intent_agent (must stay in sync)
HIGH_THRESHOLD = 0.8
LOW_THRESHOLD = 0.6


def intent_replay(initial_intent: str, steps: List[str]) -> List[Dict[str, Any]]:
    """
    Replay intent updates over the instruction sequence.
    Returns [{step: i, goal: str, version: int, update_type: str}, ...] for steps 0..N.
    Step 0 uses intent_factory(initial_intent); steps i >= 1 use sim(goal, instruction)
    with the same branching as the intent agent (refinement / major_shift / high_drift_risk).
    """
    # Lazy import to avoid circular dependency (intent may be loaded before metrics from some entry points)
    from intent import Intent, intent_factory

    if not steps:
        intent = intent_factory(initial_intent)
        return [{"step": 0, "goal": intent.goal, "version": intent.version, "update_type": "initial"}]

    intent = intent_factory(initial_intent)
    result = [{"step": 0, "goal": intent.goal, "version": intent.version, "update_type": "initial"}]

    for i in range(1, len(steps)):
        instruction = steps[i]
        emb_goal = embedding(intent.goal)
        emb_inst = embedding(instruction)
        sim = cosine_similarity(emb_goal, emb_inst)

        if sim > HIGH_THRESHOLD:
            update_type = "refinement"
            # Keep intent unchanged
            next_goal = intent.goal
            next_version = intent.version
            next_intent = intent
        elif sim < LOW_THRESHOLD:
            update_type = "major_shift"
            next_intent = intent.with_goal_replacement(
                instruction,
                update_history_append={"step": i, "old_goal": intent.goal, "new_goal": instruction, "reason": "major_shift"},
            )
            next_goal = next_intent.goal
            next_version = next_intent.version
        else:
            update_type = "high_drift_risk"
            next_intent = intent.with_goal_replacement(
                instruction,
                update_history_append={"step": i, "old_goal": intent.goal, "new_goal": instruction, "reason": "high_drift_risk"},
            )
            next_goal = next_intent.goal
            next_version = next_intent.version

        result.append({"step": i, "goal": next_goal, "version": next_version, "update_type": update_type})
        intent = next_intent

    return result
