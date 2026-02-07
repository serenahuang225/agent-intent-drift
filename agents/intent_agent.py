"""
Intent Fusion Agent: uses Intent as a first-class object (single source of truth).
The full Intent is sent to the model every time. At each step: check instruction vs intent;
update intent if allowed; log IDS.
"""
import json
import os
from pathlib import Path
from typing import Any

from metrics.ids import compute_ids, embedding, cosine_similarity

from .baseline_agent import (
    DEFAULT_MODEL,
    SEED,
    _get_model_and_tokenizer,
    run_baseline_step,
)

# Intent is first-class; when run from intent_drift_experiment, "intent" resolves to ./intent.py
from intent import Intent, intent_factory

# Similarity thresholds for controlled expansion (dynamic intent updating)
LOW_THRESHOLD = 0.30  # Below: conflicting drift (e.g. "turn the plan into a poem") — keep intent, apply constraints
HIGH_THRESHOLD = 0.75  # Above: valid elaboration — update intent to include it
# Between LOW and HIGH: treat as elaboration too — sub-tasks often have moderate similarity to goal


def _load_intent_prompt_template() -> str:
    """Load template that receives {intent_block} — intent is sent every time."""
    path = Path(__file__).resolve().parent.parent / "prompts" / "agent_with_intent.txt"
    return path.read_text(encoding="utf-8").strip()


def intent_prompt_with_object(intent: Intent, conflict_handling: str = "") -> str:
    """Build system prompt with the full Intent object (sent every single time)."""
    template = _load_intent_prompt_template()
    return template.format(intent_block=intent.to_prompt_block(), conflict_handling=conflict_handling or "")


def run_intent_step(
    model,
    tokenizer,
    intent: Intent,
    conversation: list[dict],
    user_message: str,
    initial_output: str,
    max_new_tokens: int = 256,
    seed: int = SEED,
) -> tuple[str, Intent, bool, bool]:
    """
    Run one step with controlled expansion (dynamic intent updating).
    Returns (response, next_intent, conflict_would_trigger, intent_elaborated).

    - similarity > HIGH_THRESHOLD: valid elaboration (e.g. "add risk assessment") — update intent
      so the agent can incorporate it; generate with expanded intent.
    - similarity < LOW_THRESHOLD: conflicting drift — keep intent unchanged, apply constraints.
    - else: ambiguous — keep intent unchanged, use constrained generation.
    """
    from transformers import set_seed
    set_seed(seed)

    proposed = user_message
    emb_orig = embedding(intent.goal)
    emb_proposed = embedding(proposed)
    sim = cosine_similarity(emb_orig, emb_proposed)

    if sim >= LOW_THRESHOLD:
        # Valid elaboration (includes ambiguous zone): update intent so agent can incorporate this instruction
        next_intent = intent.with_elaboration(proposed)
        conflict_would_trigger = False
        intent_elaborated = True
    else:
        # Conflicting drift (sim < LOW): keep current intent, add format-flexibility note
        next_intent = intent
        conflict_would_trigger = True
        intent_elaborated = False

    conflict_note = (
        "\n\n**IMPORTANT:** The instruction below may request a different format (e.g. bullets, "
        "table, timeline, two sentences). Follow that format while staying aligned with the "
        "overall goal. Do not refuse the request."
    ) if conflict_would_trigger else ""
    system_prompt = intent_prompt_with_object(next_intent, conflict_handling=conflict_note)
    response = run_baseline_step(
        model, tokenizer, system_prompt, conversation, user_message, max_new_tokens, seed=seed
    )
    return response, next_intent, conflict_would_trigger, intent_elaborated


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
        """Run full task with Intent object; compute IDS at every step. initial_context = article/source when present."""
        self._ensure_model()
        conversation: list[dict[str, str]] = []
        intent = intent_factory(initial_intent)
        step_logs = []
        num_steps = len(steps) if steps else 1
        if verbose:
            print("  [Intent] Task {}: {} steps (computing IDS per step)".format(task_id, num_steps))

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
        ids_0 = 0.0
        step_logs.append({
            "agent": "intent_fusion",
            "task_id": task_id,
            "step": 0,
            "intent": intent.to_dict(),
            "intent_id": intent.intent_id,
            "intent_version": intent.version,
            "intent_goal": intent.goal,
            "prompt": first_instruction,
            "output": initial_output,
            "ids": ids_0,
        })

        for i, step_instruction in enumerate(steps[1:], start=1):
            if verbose:
                print("  step {}/{} (IDS) ...".format(i, num_steps))
            response, intent, conflict, elaborated = run_intent_step(
                self._model,
                self._tokenizer,
                intent,
                conversation,
                step_instruction,
                initial_output,
                seed=seed,
            )
            ids_t = compute_ids(initial_output, response)
            step_logs.append({
                "agent": "intent_fusion",
                "task_id": task_id,
                "step": i,
                "intent": intent.to_dict(),
                "intent_id": intent.intent_id,
                "intent_version": intent.version,
                "intent_goal": intent.goal,
                "prompt": step_instruction,
                "output": response,
                "ids": round(ids_t, 4),
                "conflict_would_trigger": conflict,
                "intent_elaborated": elaborated,
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
