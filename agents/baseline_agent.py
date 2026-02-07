"""
Baseline agent: no intent fusion engine.
Uses only conversation history; no checking against original intent.

Uses Llama-3.1-8B-Instruct on CUDA: instruction-tuned, deterministic (do_sample=False).
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List

# For IDS in logs (same as intent_fusion so we can plot IDS vs step for both)
try:
    from metrics.ids import compute_ids
except ImportError:
    compute_ids = None

# Llama-3.1-8B-Instruct: instruction-tuned, runs on CUDA (gated; use HF_TOKEN or huggingface-cli login)
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SEED = 42
# Llama 3.1 has 128K context; 8192 allows full articles + long conversations
MAX_MODEL_LENGTH = 8192
# Deterministic generation: do_sample=False so randomness doesn't confuse drift results.
# Set to True to print prompt length and truncation info (quick test)
DEBUG_PROMPT = False


def _get_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Llama-3.1-8B-Instruct but is not available.")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import get_token
            hf_token = get_token()
        except Exception:
            pass
    print("[Baseline] Loading model: {} ...".format(model_name))
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    model = model.to(device)
    model.eval()  # important for consistency
    print("[Baseline] Model loaded on {}.".format(device))
    return model, tokenizer


def _build_chat_messages(system_prompt: str, conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Build messages list for chat template (user/assistant). Avoids system role for template compatibility."""
    messages = []
    system_prepended = False
    for m in conversation:
        role = m["role"]
        content = m["content"]
        if not system_prepended and role == "user":
            content = (system_prompt.strip() + "\n\n" + content).strip()
            system_prepended = True
        messages.append({"role": role, "content": content})
    return messages


def _tokenize_chat(tokenizer, messages: List[Dict[str, str]], device) -> dict:
    """Apply chat template and return tokenized inputs on device."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def _is_garbage_response(text: str) -> bool:
    """True if response is only dashes, punctuation, or whitespace (model glitch)."""
    t = text.strip()
    if not t:
        return True
    allowed = set(" -\t\n\r.,;:!?-_=*#/\\")
    return all(c in allowed for c in t) and len(t.replace(" ", "").replace("-", "").replace(".", "")) < 5


def _collapse_repetition(text: str, max_repeat: int = 2) -> str:
    """Collapse same-sentence repetition (e.g. 'X. X. X.' -> 'X. X.')."""
    if not text or len(text) < 20:
        return text
    parts = text.replace("\n", " ").split(".")
    sentences = [s.strip() for s in parts if s.strip()]
    if len(sentences) < 3:
        return text
    out = []
    prev, count = None, 0
    for s in sentences:
        if s == prev:
            count += 1
            if count <= max_repeat:
                out.append(s)
        else:
            prev, count = s, 1
            out.append(s)
    result = ". ".join(out)
    if result and not result.endswith("."):
        result += "."
    return result


def run_baseline_step(
    model,
    tokenizer,
    system_prompt: str,
    conversation: List[Dict[str, str]],
    user_message: str,
    max_new_tokens: int = 100,
    step_label: str = "",
    seed: int = SEED,
) -> str:
    """Run one step: append user message, generate reply. Deterministic (do_sample=False)."""
    import torch
    from transformers import set_seed

    if step_label:
        print("  {}".format(step_label))
    set_seed(seed)
    conversation.append({"role": "user", "content": user_message})

    device = next(model.parameters()).device
    max_input_length = MAX_MODEL_LENGTH - max_new_tokens

    # Build messages; use reduced format for step 1+ when conversation is long
    if len(conversation) >= 3:
        last_assistant_response = None
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i]["role"] == "assistant":
                last_assistant_response = conversation[i]["content"]
                break

        # Retain source content (article) from first message when collapsing — otherwise
        # the model loses the document and cannot answer summarization questions.
        content_header = "Content to work with:"
        first_user_content = conversation[0]["content"] if conversation else ""
        retained_context = ""
        if content_header in first_user_content:
            # Extract "Content to work with:\n\n[article]\n\n" — keep article for later steps
            idx = first_user_content.find(content_header)
            rest = first_user_content[idx + len(content_header) :].strip()
            # Stop at "Instruction:" or "Your response:" to get just the article
            for sep in ["\n\nInstruction:", "\n\nYour response:"]:
                if sep in rest:
                    rest = rest.split(sep)[0].strip()
            if rest:
                max_context_chars = 4000  # keep enough for typical abstracts/articles
                if len(rest) > max_context_chars:
                    rest = rest[:max_context_chars] + "\n...[truncated]"
                retained_context = content_header + "\n\n" + rest + "\n\n"

        if last_assistant_response:
            combined_user = (
                retained_context
                + "Current instruction: "
                + user_message
                + "\n\nPrevious response to refine:\n"
                + last_assistant_response
            )
        else:
            first_user_content = conversation[0]["content"]
            combined_user = first_user_content + "\n\n" + user_message

        messages = [
            {"role": "user", "content": (system_prompt.strip() + "\n\n" + combined_user).strip()},
        ]
    else:
        messages = _build_chat_messages(system_prompt, conversation)

    inputs = _tokenize_chat(tokenizer, messages, device)
    seq_len = inputs["input_ids"].shape[1]

    if seq_len > max_input_length and len(conversation) >= 3 and last_assistant_response:
        # Truncate the previous response from the left to fit context; keep retained_context if present
        max_prev_chars = 5000  # approximate; we trim and retry
        for _ in range(5):
            if len(last_assistant_response) > max_prev_chars:
                truncated_prev = "..." + last_assistant_response[-max_prev_chars:]
            else:
                truncated_prev = last_assistant_response
            combined_user = (
                retained_context
                + "Current instruction: "
                + user_message
                + "\n\nPrevious response to refine:\n"
                + truncated_prev
            )
            messages = [
                {"role": "user", "content": (system_prompt.strip() + "\n\n" + combined_user).strip()},
            ]
            inputs = _tokenize_chat(tokenizer, messages, device)
            seq_len = inputs["input_ids"].shape[1]
            if seq_len <= max_input_length:
                break
            max_prev_chars = int(max_prev_chars * 0.6)

    if seq_len > max_input_length:
        # Truncate from the left: keep the last max_input_length tokens
        ids = inputs["input_ids"][0]
        inputs["input_ids"] = ids[-max_input_length:].unsqueeze(0)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][0][-max_input_length:].unsqueeze(0)
        seq_len = max_input_length

    if DEBUG_PROMPT:
        print("  [debug] Prompt length: {} (max {})".format(seq_len, max_input_length))

    input_len = inputs["input_ids"].shape[1]
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic = good for experiments
            pad_token_id=pad_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    # Decode only the newly generated tokens (robust: no prefix-length mismatch)
    new_token_ids = out[0][input_len:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    # Trim at turn boundaries or spillover into next turn
    for sep in ["<|im_start|>", "<|im_end|>", "<|eot_id|>", "<|end_header_id|>", "\nUser:", "\nAssistant:"]:
        if sep in response:
            response = response.split(sep)[0].strip()
    response = _collapse_repetition(response)
    # Treat dash-only or punctuation-only output as empty (model glitch)
    if _is_garbage_response(response) and len(conversation) >= 2:
        response = ""
    if not response.strip() and len(conversation) >= 2:
        for i in range(len(conversation) - 1, -1, -1):
            if conversation[i]["role"] == "assistant":
                response = conversation[i]["content"]
                break
    conversation.append({"role": "assistant", "content": response})
    return response


class BaselineAgent:
    """Standard LLM agent with no explicit intent representation."""

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
            self._model, self._tokenizer = _get_model_and_tokenizer(self.model_name)

    def run_task(
        self,
        task_id: str,
        initial_intent: str,
        steps: List[str],
        system_prompt: str,
        initial_context: str = "",
        verbose: bool = True,
        seed: int = SEED,
    ) -> List[Dict[str, Any]]:
        """Run full task; return list of step logs. initial_context (e.g. article) is shown with the first step."""
        self._ensure_model()
        conversation: List[Dict[str, str]] = []
        step_logs = []
        num_steps = len(steps) if steps else 1

        if verbose:
            print("  [Baseline] Task {}: {} steps".format(task_id, num_steps))

        # First user message: make article/source explicit so the model knows what to summarize or work with
        first_instruction = steps[0] if steps else initial_intent
        if initial_context and first_instruction:
            first_user_msg = (
                "Content to work with:\n\n" + initial_context.strip()
                + "\n\nInstruction: " + first_instruction
                + "\n\nYour response:"
            )
        else:
            first_user_msg = first_instruction

        # Step 0: initial instruction (with context) → get initial output
        initial_output = run_baseline_step(
            self._model,
            self._tokenizer,
            system_prompt,
            conversation,
            first_user_msg,
            step_label="  step 0/{} ...".format(num_steps) if verbose else "",
            seed=seed,
        )

        step_logs.append({
            "agent": "baseline",
            "task_id": task_id,
            "step": 0,
            "prompt": first_instruction,
            "output": initial_output,
            "ids": 0.0,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

        for i, step_instruction in enumerate(steps[1:], start=1):
            out = run_baseline_step(
                self._model,
                self._tokenizer,
                system_prompt,
                conversation,
                step_instruction,
                step_label="  step {}/{} ...".format(i, num_steps) if verbose else "",
                seed=seed,
            )
            ids_t = round(compute_ids(initial_output, out), 4) if compute_ids else None
            log_entry = {
                "agent": "baseline",
                "task_id": task_id,
                "step": i,
                "prompt": step_instruction,
                "output": out,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            if ids_t is not None:
                log_entry["ids"] = ids_t
            step_logs.append(log_entry)

        if verbose:
            print("  [Baseline] Task {} done.".format(task_id))
        return step_logs

    def run_and_log(
        self,
        task_id: str,
        initial_intent: str,
        steps: List[str],
        system_prompt: str,
        initial_context: str = "",
        log_file: str = "baseline_runs.jsonl",
        verbose: bool = True,
        seed: int = SEED,
    ) -> List[Dict[str, Any]]:
        """Run task and append each step log to log_file. initial_context = article/source text when present."""
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, log_file)
        step_logs = self.run_task(
            task_id, initial_intent, steps, system_prompt,
            initial_context=initial_context, verbose=verbose, seed=seed,
        )
        with open(path, "a", encoding="utf-8") as f:
            for log in step_logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
        return step_logs
