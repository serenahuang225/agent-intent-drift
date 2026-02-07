# 📁 PROJECT OVERVIEW

**Project Name:** Measuring Agent Intent Preservation
**Goal:** Compare a baseline LLM agent vs. an intent-anchored agent and measure semantic intent drift using an Intent Drift Score (IDS).

**Core Experimental Variable:** Presence vs. absence of an Intent Fusion Engine
**Controlled Variables:** Same model, same prompts, same tasks, same seed
**Measured Outcome:** Intent Drift Score over time

---

# 1️⃣ ENVIRONMENT & SETUP

### Model

* Use **one open-source LLM** for all experiments
* Recommended:

  * `meta-llama/Llama-3-8B-Instruct` (GPU) ***
  * or `mistralai/Mistral-7B-Instruct`
* Temperature fixed (e.g. `0.7`)
* Seed fixed for repeatability

### Libraries

```txt
transformers
torch
sentence-transformers
numpy
pandas
scikit-learn
matplotlib
```

### Directory Structure

```txt
intent_drift_experiment/
│
├── prompts/
│   ├── agent_base.txt
│   ├── agent_with_intent.txt
│
├── tasks/
│   ├── summarization_tasks.json
│   ├── planning_tasks.json
│
├── logs/
│   ├── baseline_runs.jsonl
│   ├── intent_runs.jsonl
│
├── metrics/
│   └── ids.py
│
├── agents/
│   ├── baseline_agent.py
│   ├── intent_agent.py
│
└── run_experiment.py
```

---

# 2️⃣ TASKS & PROMPTS (REPEATABLE INPUTS)

### Task Types (Fixed Set)

Use **exact same tasks** for both agents.

#### A. Summarization Chain

Example structure in `summarization_tasks.json`:

```json
[
  {
    "initial_intent": "Summarize the article clearly and concisely.",
    "steps": [
      "Summarize the article.",
      "Make it shorter.",
      "Add more detail.",
      "Rewrite in a more analytical tone.",
      "Make it engaging for a general audience."
    ]
  }
]
```

#### B. Planning Task

```json
[
  {
    "initial_intent": "Create a step-by-step plan to study for a math exam.",
    "steps": [
      "Create a study plan.",
      "Make it more detailed.",
      "Optimize it for limited time.",
      "Rewrite it to be motivational.",
      "Add advanced strategies."
    ]
  }
]
```

⚠️ **Do not change tasks between runs.**

---

# 3️⃣ BASELINE AGENT (NO INTENT FUSION ENGINE)

### Purpose

Simulates a standard LLM agent that:

* Uses only conversation history
* Has no explicit intent representation
* Is vulnerable to silent drift

### Prompt (`agent_base.txt`)

```txt
You are a helpful AI assistant.
Follow the user's instructions carefully.
```

### Behavior

* Each step appends to conversation history
* No checking against original intent
* No semantic anchoring

### Output Logging

For each step, log:

```json
{
  "agent": "baseline",
  "task_id": "...",
  "step": 3,
  "prompt": "...",
  "output": "...",
  "timestamp": "..."
}
```

---

# 4️⃣ INTENT DRIFT SCORE (IDS)

### Definition

IDS measures **semantic divergence from the initial intent output**.

### Embedding Model

Use:

```python
sentence-transformers/all-MiniLM-L6-v2
```

### IDS Formula

```python
IDS_t = 1 - cosine_similarity(
    embedding(initial_output),
    embedding(current_output)
)
```

* IDS = 0 → perfectly aligned
* IDS → 1 → maximum semantic drift

### Implementation Rules

* Compute IDS **at every step**
* Initial step IDS = 0
* Store IDS per step per run

---

# 5️⃣ INTENT FUSION ENGINE (CORE CONTRIBUTION)

### Intent State Object

```python
intent_state = {
    "goal": "Summarize the article clearly and concisely.",
    "version": 1,
    "confidence": 1.0,
    "last_confirmed_step": 0
}
```

---

### Intent Update Rule (Semantic Commit–Inspired)

At each step:

1. Extract **proposed intent change** from the new instruction
2. Compute semantic similarity between:

   * original intent
   * proposed intent
3. Apply update logic:

```python
if similarity < INTENT_SIM_THRESHOLD:
    require_confirmation = True
else:
    update_intent_state()
```

For the experiment:

* **Auto-confirm** (to keep runs deterministic)
* Log when a conflict would have been triggered

---

# 6️⃣ AGENT WITH INTENT FUSION ENGINE

### Prompt (`agent_with_intent.txt`)

```txt
You are a helpful AI assistant.
You must follow the CURRENT INTENT exactly.

CURRENT INTENT:
{intent_state.goal}

If an instruction conflicts with the intent, prioritize the intent.
```

### Execution Loop

For each step:

1. Check instruction vs. intent
2. Decide:

   * aligned → proceed
   * conflicting → constrain output to original intent
3. Generate response
4. Update intent version if allowed
5. Log everything

### Output Logging

```json
{
  "agent": "intent_fusion",
  "task_id": "...",
  "step": 3,
  "intent_version": 1,
  "intent_goal": "...",
  "prompt": "...",
  "output": "...",
  "ids": 0.12
}
```

---

# 7️⃣ EXPERIMENT EXECUTION (REPEATABLE)

### Run Conditions

* Same model
* Same seed
* Same temperature
* Same tasks
* Run each task **N = 5 times**

### Execution Order

1. Run **baseline agent**
2. Run **intent fusion agent**
3. Store logs separately

---

# 8️⃣ ANALYSIS & EXPECTED RESULT

### Compute:

* Mean IDS per step
* IDS trajectory over time
* Final IDS

### Expected Outcome

```txt
IDS_baseline  >  IDS_intent_fusion
```

Plot:

* IDS vs. step number
* Two lines: baseline vs. intent fusion

---

# 9️⃣ WHAT THIS EXPERIMENT DEMONSTRATES

* Intent drift occurs even when outputs appear fluent
* Explicit semantic intent anchoring reduces drift
* IDS captures alignment loss missed by general stability metrics
* Method works across any LLM (model-agnostic)

---

# 10️⃣ FINAL NOTE TO CURSOR

> This project must be deterministic, fully logged, and reproducible.
> Do NOT use external APIs or non-deterministic sampling.
> All randomness must be seeded.

---

## ✅ This Is a Legit Research-Grade Experiment

* Clear independent variable
* Clear dependent variable
* Grounded in current literature
* Novel framing
* Feasible implementation
