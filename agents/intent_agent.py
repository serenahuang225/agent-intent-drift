"""
Intent Fusion Agent: uses Intent as a first-class object (single source of truth).
The full Intent is sent to the model every time. At each step: check instruction vs intent;
update intent via with_goal_replacement when sim is low; log update_type (USD computed at interpretation).
"""
import json
import os
from pathlib import Path
from typing import Any, Literal

from metrics.ids import embedding, cosine_similarity

from .baseline_agent import (
    DEFAULT_MODEL,
    SEED,
    _get_model_and_tokenizer,
    run_baseline_step,
)

# Intent is first-class; when run from intent_drift_experiment, "intent" resolves to ./intent.py
from intent import Intent, intent_factory

# Similarity thresholds: sim > HIGH → keep; sim < LOW → major shift; else high_drift_risk (update + log)
HIGH_THRESHOLD = 0.8
LOW_THRESHOLD = 0.6

UpdateType = Literal["refinement", "major_shift", "high_drift_risk"]


def _load_intent_prompt_template() -> str:
    """Load template that receives {intent_block} — intent is sent every time."""
    path = Path(__file__).resolve().parent.parent / "prompts" / "agent_with_intent.txt"
    return path.read_text(encoding="utf-8").strip()


def intent_prompt_with_object(intent: Intent) -> str:
    """Build system prompt with the full Intent object (sent every single time). No conflict_handling."""
    template = _load_intent_prompt_template()
    return template.format(intent_block=intent.to_prompt_block())


def run_intent_step(
    model,
    tokenizer,
    intent: Intent,
    conversation: list[dict],
    user_message: str,
    max_new_tokens: int = 256,
    seed: int = SEED,
) -> tuple[str, Intent, UpdateType]:
    """
    Run one step with inverted intent engine.
    Returns (response, next_intent, update_type).

    - sim > 0.8: Refinement. Keep intent, no update.
    - sim < 0.6: Major goal shift. with_goal_replacement(instruction), reason "major_shift".
    - 0.6 <= sim <= 0.8: Ambiguous. with_goal_replacement(instruction), reason "high_drift_risk".
    Always send current intent to the model (no blocking).
    """
    from transformers import set_seed
    set_seed(seed)

    instruction = user_message
    emb_goal = embedding(intent.goal)
    emb_inst = embedding(instruction)
    sim = cosine_similarity(emb_goal, emb_inst)

    if sim > HIGH_THRESHOLD:
        update_type: UpdateType = "refinement"
        next_intent = intent
    elif sim < LOW_THRESHOLD:
        update_type = "major_shift"
        next_intent = intent.with_goal_replacement(
            instruction,
            update_history_append={
                "old_goal": intent.goal,
                "new_goal": instruction,
                "reason": "major_shift",
            },
        )
    else:
        update_type = "high_drift_risk"
        # Elaborate rather than replace: keep previous focus and add this step (reduces losing user's thread)
        next_intent = intent.with_elaboration(instruction)
        next_intent = Intent(
            intent_id=next_intent.intent_id,
            goal=next_intent.goal,
            constraints=next_intent.constraints,
            success_criteria=next_intent.success_criteria,
            assumptions=next_intent.assumptions,
            confidence=next_intent.confidence,
            last_confirmed=next_intent.last_confirmed,
            version=next_intent.version,
            update_history=next_intent.update_history
            + [{"old_goal": intent.goal, "new_goal": next_intent.goal, "reason": "high_drift_risk"}],
            overall_goal=next_intent.overall_goal,
        )

    system_prompt = intent_prompt_with_object(next_intent)
    response = run_baseline_step(
        model, tokenizer, system_prompt, conversation, user_message, max_new_tokens, seed=seed
    )
    return response, next_intent, update_type


class IntentAgent:
    """Agent with Intent Fusion Engine; Intent is first-class and sent every turn."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        log_dir: str = "logs",
        seed: int = SEED,
    ):
        self.model_name = model_name
        self.log_dir = log_dir
        self.seed = seed
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is None:
            print("[Intent] Loading model: {} ...".format(self.model_name))
            self._model, self._tokenizer = _get_model_and_tokenizer(self.model_name)
            print("[Intent] Model loaded.")

    def run_task(
        self,
        task_id: str,
        initial_intent: str,
        steps: list[str],
        initial_context: str = "",
        verbose: bool = True,
        seed: int = SEED,
    ) -> list[dict[str, Any]]:
        """Run full task with Intent object; log update_type and update_history (USD computed at interpretation)."""
        self._ensure_model()
        conversation: list[dict[str, str]] = []
        intent = intent_factory(initial_intent)
        step_logs = []
        num_steps = len(steps) if steps else 1
        if verbose:
            print("  [Intent] Task {}: {} steps".format(task_id, num_steps))

        first_instruction = steps[0] if steps else initial_intent
        if initial_context and first_instruction:
            first_user_msg = (
                "Content to work with:\n\n" + initial_context.strip()
                + "\n\nInstruction: " + first_instruction
                + "\n\nYour response:"
            )
        else:
            first_user_msg = first_instruction

        # Step 0 — prompt includes full Intent object
        system_prompt = intent_prompt_with_object(intent)
        if verbose:
            print("  step 0/{} ...".format(num_steps))
        initial_output = run_baseline_step(
            self._model,
            self._tokenizer,
            system_prompt,
            conversation,
            first_user_msg,
            seed=seed,
        )
        step_logs.append({
            "agent": "intent_fusion",
            "task_id": task_id,
            "step": 0,
            "intent": intent.to_dict(),
            "intent_id": intent.intent_id,
            "intent_version": intent.version,
            "intent_goal": intent.goal,
            "update_type": "initial",
            "update_history": list(intent.update_history),
            "prompt": first_instruction,
            "output": initial_output,
        })

        for i, step_instruction in enumerate(steps[1:], start=1):
            if verbose:
                print("  step {}/{} ...".format(i, num_steps))
            response, intent, update_type = run_intent_step(
                self._model,
                self._tokenizer,
                intent,
                conversation,
                step_instruction,
                seed=seed,
            )
            step_logs.append({
                "agent": "intent_fusion",
                "task_id": task_id,
                "step": i,
                "intent": intent.to_dict(),
                "intent_id": intent.intent_id,
                "intent_version": intent.version,
                "intent_goal": intent.goal,
                "update_type": update_type,
                "update_history": list(intent.update_history),
                "prompt": step_instruction,
                "output": response,
            })

        if verbose:
            print("  [Intent] Task {} done.".format(task_id))
        return step_logs

    def run_and_log(
        self,
        task_id: str,
        initial_intent: str,
        steps: list[str],
        initial_context: str = "",
        log_file: str = "intent_runs.jsonl",
        verbose: bool = True,
        seed: int = SEED,
    ) -> list[dict[str, Any]]:
        """Run task and append each step log to log_file."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, log_file)
        step_logs = self.run_task(
            task_id, initial_intent, steps,
            initial_context=initial_context, verbose=verbose, seed=seed,
        )
        with open(path, "a", encoding="utf-8") as f:
            for log in step_logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
        return step_logs
