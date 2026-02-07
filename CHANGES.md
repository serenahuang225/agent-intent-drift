# Experiment Changes: Summary and Code Comparisons

This document summarizes all changes made to the intent drift experiment: multi-model runs, interpretation and visualizations, download script, intent object improvements, and controlled expansion (dynamic intent updating). Each section includes **before vs after** comparisons and the rationale.

---

## 1. Multi-Model Experiment (`run_experiment.py`)

### Previous behavior
- Single model only: `meta-llama/Llama-3.1-8B-Instruct`.
- One pair of log files: `baseline_runs.jsonl`, `intent_runs.jsonl`.
- Agents created once; baseline then intent run for all tasks; GPU freed once between baseline and intent.

### New behavior
- **Multiple models** run in sequence (Llama-3.1-8B, Gemma-2-2b-it, Mistral-7B-Instruct), each with its own log files.
- **Per-model logs**: `baseline_runs_<slug>.jsonl`, `intent_runs_<slug>.jsonl` (e.g. `baseline_runs_llama-3.1-8b-instruct.jsonl`).
- **GPU freed between models** so each model loads in a clean state.
- **CLI**: `--models model1,model2` overrides the default model list.

### Code comparison

**Before (single model, single log):**
```python
# No MODELS list; agents created once
baseline_agent = BaselineAgent(log_dir=str(LOGS_DIR), seed=SEED)
intent_agent = IntentAgent(log_dir=str(LOGS_DIR), seed=SEED)
# ...
baseline_agent.run_and_log(..., log_file="baseline_runs.jsonl")
# ... free GPU ...
intent_agent.run_and_log(..., log_file="intent_runs.jsonl")
interpret_results(LOGS_DIR)
```

**After (multi-model, per-model logs):**
```python
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

def model_to_slug(model_name: str) -> str:
    """Filesystem-safe slug from model name."""
    name = model_name.split("/")[-1] if "/" in model_name else model_name
    slug = "".join(c if c.isalnum() or c in "-." else "_" for c in name.lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")

# In main():
for model in models_to_run:
    slug = model_to_slug(model)
    baseline_log = "baseline_runs_{}.jsonl".format(slug)
    intent_log = "intent_runs_{}.jsonl".format(slug)
    baseline_agent = BaselineAgent(log_dir=str(LOGS_DIR), model_name=model, seed=SEED)
    intent_agent = IntentAgent(log_dir=str(LOGS_DIR), model_name=model, seed=SEED)
    # ... run all tasks with baseline_log / intent_log ...
    # Free GPU before next model
    intent_agent._model = None
    intent_agent._tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()

interpret_all_models(LOGS_DIR, models_to_run)
```

### Improvements
- Experiment is **more rigorous** across three models (8B, 2B, 7B) that fit RTX 3090.
- Results are **comparable per model** and not overwritten.
- **Reproducibility**: same seeds and tasks; model list and slugs are explicit.

---

## 2. Interpretation and Visualizations (`run_experiment.py`)

### Previous behavior
- `interpret_results(logs_dir)` only: read `baseline_runs.jsonl` and `intent_runs.jsonl` from `logs_dir`, write all outputs (CSV, PNG, report) into `logs_dir`.

### New behavior
- **Generalized** `interpret_results(logs_dir, baseline_path=None, intent_path=None, output_subdir=None)`.
  - When paths are given, they are used; otherwise defaults under `logs_dir`.
  - When `output_subdir` is set, all outputs go to `logs_dir / output_subdir` (per-model subdirs).
- **Per-model interpretation**: each model gets its own folder (e.g. `logs/llama-3.1-8b-instruct/`) with `task_comparison.csv`, `summary_stats.csv`, `ids_by_step.png`, `ids_by_task_type.png`, `ids_per_task.png`, `experiment_report.md`.
- **Cross-model visualizations** (after all per-model runs):
  - `logs/ids_by_model.png`: bar chart of mean IDS (baseline vs intent) per model.
  - `logs/ids_by_step_all_models.png`: intent fusion mean IDS by step, one curve per model.
  - `logs/cross_model_summary.csv`: model_slug, baseline_mean_ids, intent_mean_ids, intent_wins, total_tasks.
- **interpret_all_models(logs_dir, models)**: runs interpretation per model then writes cross-model artifacts; falls back to default log names if no per-model logs exist.

### Code comparison

**Before:**
```python
def interpret_results(logs_dir: Path) -> bool:
    baseline_log = logs_dir / "baseline_runs.jsonl"
    intent_log = logs_dir / "intent_runs.jsonl"
    # ... compute task_rows, aggregate stats ...
    task_csv = logs_dir / "task_comparison.csv"
    fig.savefig(logs_dir / "ids_by_step.png", ...)
    (logs_dir / "experiment_report.md").write_text(...)
```

**After:**
```python
def interpret_results(logs_dir: Path, baseline_path=None, intent_path=None, output_subdir=None) -> bool:
    if baseline_path is None:
        baseline_path = logs_dir / "baseline_runs.jsonl"
    if intent_path is None:
        intent_path = logs_dir / "intent_runs.jsonl"
    out_dir = (logs_dir / output_subdir) if output_subdir else logs_dir
    if output_subdir:
        out_dir.mkdir(parents=True, exist_ok=True)
    # ... same logic but write to out_dir ...
    task_csv = out_dir / "task_comparison.csv"
    fig.savefig(out_dir / "ids_by_step.png", ...)

def interpret_all_models(logs_dir: Path, models: list) -> bool:
    for slug in [model_to_slug(m) for m in models]:
        baseline_path = logs_dir / "baseline_runs_{}.jsonl".format(slug)
        intent_path = logs_dir / "intent_runs_{}.jsonl".format(slug)
        interpret_results(logs_dir, baseline_path=baseline_path, intent_path=intent_path, output_subdir=slug)
    _write_cross_model_visualizations(logs_dir, per_model_results)
```

### Improvements
- **Per-model** results are isolated and comparable.
- **Cross-model** view supports “which model preserves intent best?” and “how does drift over steps differ by model?”.
- **Backward compatible**: if only default log files exist, interpretation still runs.

---

## 3. Download Script (`download_model.py`)

### Previous behavior
- Single causal model: Llama-3.1-8B-Instruct; one function `download_causal_model()` with no arguments.
- No cache check: every run re-downloaded (or re-used cache implicitly only when loading).

### New behavior
- **CAUSAL_MODELS** list aligned with `run_experiment.MODELS` (Llama, Gemma-2-2b-it, Mistral-7B-Instruct).
- **Cache check**: `_model_cached(repo_id)` uses `huggingface_hub.snapshot_download(..., local_files_only=True)`; if the repo is already cached, no network call and no model load.
- **Skip when cached**: `download_causal_model(model_name)` and `download_embedding_model()` print “already in cache, skipping download” and return without downloading when cached.

### Code comparison

**Before:**
```python
def download_causal_model():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    # ...
    tokenizer = AutoTokenizer.from_pretrained(model_name, ...)
    model = AutoModelForCausalLM.from_pretrained(model_name, ...)
```

**After:**
```python
CAUSAL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

def _model_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False

def download_causal_model(model_name: str):
    if _model_cached(model_name):
        print(f"  {model_name} already in cache, skipping download.")
        return None, None
    # ... download as before ...
```

### Improvements
- **No redundant downloads** when models are already cached.
- **One script** prepares all models used by the experiment; cache check is lightweight (no full model load).

---

## 4. Intent Object and Prompt (`intent.py`, `prompts/agent_with_intent.txt`)

### 4.1 `to_prompt_block()` — More directive, non-ignorable

**Before (compact, minimal):**
```python
def to_prompt_block(self) -> str:
    lines = ["goal: " + self.goal, "version: " + str(self.version)]
    if self.constraints:
        lines.append("constraints: " + ", ".join(self.constraints))
    if self.success_criteria:
        lines.append("success_criteria: " + "; ".join(self.success_criteria))
    if self.assumptions:
        lines.append("assumptions: " + "; ".join(self.assumptions))
    return "\n".join(lines)
```

**After (directive, with explicit verification step):**
```python
def to_prompt_block(self) -> str:
    prompt_lines = [
        "### ACTIVE USER INTENT (You MUST follow this) ###",
        "**Primary Goal:** " + self.goal,
        "**Version:** " + str(self.version),
    ]
    if self.constraints:
        prompt_lines.append("**Constraints:** " + "; ".join("'{}'".format(c) for c in self.constraints))
    if self.success_criteria:
        prompt_lines.append("**Success Looks Like:** " + "; ".join("'{}'".format(s) for s in self.success_criteria))
    if self.assumptions:
        prompt_lines.append("**Assumptions:** " + "; ".join(self.assumptions))
    prompt_lines.append(
        "\n**INSTRUCTION:** Before finalizing your response, YOU MUST explicitly verify it aligns with the Primary Goal and all Constraints above."
    )
    return "\n".join(prompt_lines)
```

**Improvement:** The intent is framed as a **required** object with a clear “verify before finalizing” step, so the model is instructed to check its answer against the goal and constraints.

---

### 4.2 Constraint derivation — More actionable rules

**Before:** A few simple rules (e.g. “one sentence” → one success criterion; “for teenagers” → one constraint).

**After:** Richer, more specific derivations so the LLM has concrete, checkable rules:

| Goal wording | Constraint / success criterion (new) |
|--------------|--------------------------------------|
| one sentence / in one sentence | RESPONSE MUST BE EXACTLY ONE GRAMMATICAL SENTENCE.; no lists/bullets/multiple clauses |
| for teenagers / younger audience | Casual language, avoid jargon; relate to school, social media, pop culture |
| non-technical / layman | Explain simply; use everyday analogies |
| under 10 words | Word count < 10; concise and impactful |
| headline | Headline style: present tense, active voice, attention-grabbing |
| balanced / pros and cons | Structure: Pros: / Cons:; at least one pro and one con |
| summarize (no “detail”) | Prioritize brevity; only most consequential information |
| comparison / compare | Comparative structure (e.g. “Whereas X…, Y…” or table) |
| recommend / should | End with clear, justified recommendation (“Therefore, I recommend…”) |

**Improvement:** Vague goals are turned into **actionable** constraints and success criteria the model can follow and the experiment can evaluate.

---

### 4.3 System prompt (`prompts/agent_with_intent.txt`)

**Before:**
```
You are a helpful AI assistant.
You must follow the CURRENT INTENT exactly. The intent is a first-class object — it is the single source of truth.
Always respond only in English.

CURRENT INTENT:
{intent_block}

If an instruction conflicts with the intent, prioritize the intent.
```

**After:**
```
You are a helpful AI assistant.
The CURRENT INTENT below is the single source of truth. It is fixed and must not be replaced by later instructions.
You must follow the goal exactly and satisfy all constraints and success_criteria. Your response must satisfy the success_criteria and constraints listed below.
If an instruction conflicts with the intent, prioritize the intent.
Always respond only in English.

CURRENT INTENT (fixed — do not drift from this):
{intent_block}
```

**Improvement:** Explicit requirement to satisfy **constraints** and **success_criteria**, and to treat the intent as **fixed** unless the engine updates it (e.g. via elaboration).

---

## 5. Intent Fusion Logic: From “Always Freeze” to “Controlled Expansion” (`intent_agent.py`, `intent.py`)

### Previous behavior (after first “preserve goal” change)
- Single threshold: below 0.5 → “conflict” (keep intent); above 0.5 the goal was **no longer** updated (intent always unchanged).
- Effect: planning tasks where steps like “Add risk assessment” are **valid elaborations** were over-constrained because the intent never expanded.

### New behavior: dynamic intent updating (controlled expansion)
- **Two thresholds:**
  - **LOW_THRESHOLD = 0.5**: Below → **conflicting drift** (e.g. “turn the plan into a poem”). Keep intent; generate under current constraints.
  - **HIGH_THRESHOLD = 0.75**: Above → **valid elaboration** (e.g. “add risk assessment”). **Update** intent to include the new requirement; generate with expanded intent.
  - **Between**: **Ambiguous**. Keep intent unchanged; constrained generation.
- **Intent expansion**: `Intent.with_elaboration(new_instruction)` appends to the goal:  
  `"... Additionally, the response must address: " + new_instruction + "."`  
  and bumps the version.
- **Logging**: Each step log has `intent_elaborated: true/false` so you can see when the intent was expanded.

### Code comparison

**Before (single threshold, intent never updated):**
```python
INTENT_SIM_THRESHOLD = 0.5

def run_intent_step(...) -> tuple[str, Intent, bool]:
    sim = cosine_similarity(emb_orig, emb_proposed)
    require_confirmation = sim < INTENT_SIM_THRESHOLD
    next_intent = intent  # always unchanged
    # ...
    return response, next_intent, require_confirmation
```

**After (high / low thresholds, elaboration updates intent):**
```python
LOW_THRESHOLD = 0.5   # Below: conflicting drift — keep intent
HIGH_THRESHOLD = 0.75 # Above: valid elaboration — update intent

def run_intent_step(...) -> tuple[str, Intent, bool, bool]:
    sim = cosine_similarity(emb_orig, emb_proposed)
    if sim > HIGH_THRESHOLD:
        next_intent = intent.with_elaboration(proposed)
        conflict_would_trigger = False
        intent_elaborated = True
    elif sim < LOW_THRESHOLD:
        next_intent = intent
        conflict_would_trigger = True
        intent_elaborated = False
    else:
        next_intent = intent
        conflict_would_trigger = False
        intent_elaborated = False
    # ... generate with next_intent ...
    return response, next_intent, conflict_would_trigger, intent_elaborated
```

**New in `intent.py`:**
```python
def with_elaboration(self, new_instruction: str) -> "Intent":
    addition = new_instruction.strip().rstrip(".")
    expanded_goal = self.goal.rstrip(". ") + ". Additionally, the response must address: " + addition + "."
    return self.with_updated_goal(expanded_goal, bump_version=True)
```

### Improvements
- **Planning tasks**: Instructions that **elaborate** the plan (high similarity) are incorporated into the intent instead of being blocked, reducing over-constraint.
- **Summarization**: Still anchored when later instructions are low-similarity drift; high-similarity refinements (e.g. “make it shorter”) can be added as elaborations.
- **Clear semantics**: “Conflicting drift” vs “valid elaboration” vs “ambiguous” is explicit and logged.

---

## 6. Diagnostic Script (`debug_intent.py`)

### Purpose
- Run **1–2 tasks** (e.g. failing planning tasks) without a full experiment.
- Print **intent block**, **instruction–goal similarity** per step, and optionally **agent output previews**.
- End with **post-run analysis questions** to identify failure patterns (weak constraint, intent overridden, format error).

### Usage
```bash
cd intent_drift_experiment
python debug_intent.py                              # default: summarization_0, planning_0
python debug_intent.py --task summarization_0        # one task
python debug_intent.py --tasks summarization_0,planning_1
python debug_intent.py --no-run                      # intent + similarities only, no model run
python debug_intent.py --model meta-llama/Llama-3.1-8B-Instruct
```

Set **FAILING_TASK_IDS** in the script to the task IDs where baseline beat intent (from `task_comparison.csv`); the script will then run the first two of those by default.

### Improvements
- **Faster iteration** on failing tasks.
- **Explicit similarity and elaboration** visibility before re-running the full experiment.

---

## 7. File-Level Summary

| File | Changes |
|------|--------|
| **run_experiment.py** | MODELS, model_to_slug, multi-model loop, per-model logs, GPU free between models; interpret_results(paths, output_subdir), interpret_all_models, cross-model visualizations; --models CLI. |
| **download_model.py** | CAUSAL_MODELS, _model_cached(), skip-if-cached in download_causal_model and download_embedding_model. |
| **intent.py** | to_prompt_block() directive format; derive_constraints_from_goal() expanded rules; with_elaboration(); intent_factory(derive_from_goal=True). |
| **agents/intent_agent.py** | LOW_THRESHOLD / HIGH_THRESHOLD; run_intent_step() three-way logic and (response, intent, conflict, intent_elaborated); step logs include intent_elaborated. |
| **prompts/agent_with_intent.txt** | Stronger wording: fixed intent, satisfy constraints/success_criteria. |
| **debug_intent.py** | New: diagnostic script for 1–2 tasks with intent block, similarities, optional run, analysis questions. |

---

## 8. Narrative for Reporting

You can describe the evolution of the experiment as follows:

1. **Multi-model rigor**: The same protocol runs over three models (Llama-3.1-8B, Gemma-2-2b-it, Mistral-7B-Instruct) with per-model and cross-model analysis.
2. **Intent as a first-class object**: The intent is surfaced in the system prompt in a **directive, non-ignorable** way, with derived **constraints** and **success criteria** so the model has concrete rules to follow.
3. **Controlled expansion**: The engine distinguishes **harmful drift** (low similarity → keep intent) from **helpful elaboration** (high similarity → update intent). This addresses the “static intent vs dynamic task” mismatch and is designed to improve performance on planning while preserving gains on summarization.
4. **Diagnostics**: The debug script supports targeted analysis of failing tasks before re-running the full experiment.

Together, these changes make the experiment more rigorous, the intent more actionable, and the fusion logic better suited to both content-refinement tasks (summarization) and creative-construction tasks (planning).
