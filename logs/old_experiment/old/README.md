# Measuring Agent Intent Preservation

This project compares a **baseline LLM agent** (no explicit intent) with an **intent-anchored agent** (Intent Fusion Engine) and measures semantic intent drift using an **Intent Drift Score (IDS)**. The experiment is deterministic, fully logged, and reproducible.

---

## Table of Contents

1. [Overview and Goal](#overview-and-goal)
2. [How the Results Were Accomplished](#how-the-results-were-accomplished)
3. [Project Layout and Every File Explained](#project-layout-and-every-file-explained)
4. [Results](#results)
5. [Environment Setup](#environment-setup)
6. [Running the Experiment](#running-the-experiment)
7. [Interpretation and Outputs](#interpretation-and-outputs)

---

## Overview and Goal

- **Goal:** Show that explicit intent anchoring (Intent Fusion) can reduce semantic drift from the user’s initial intent compared to a baseline agent that only follows conversation history.
- **Independent variable:** Presence vs. absence of the Intent Fusion Engine (baseline vs. intent agent).
- **Controlled variables:** Same LLM (Qwen2.5-7B-Instruct), same seed, same tasks, deterministic generation (`do_sample=False`).
- **Dependent variable:** Intent Drift Score (IDS) at each step—lower means more aligned with the initial output (and thus initial intent).

**Intent Drift Score (IDS):**

- **IDS = 0** → output is semantically aligned with the initial response (no drift).
- **IDS → 1** → maximum semantic drift from the initial response.
- **Lower IDS is better** for intent preservation.

---

## How the Results Were Accomplished

### End-to-end flow

1. **Seed and environment**
   - Global seed `42` is set for `random`, `numpy`, and `torch` so runs are reproducible.

2. **Load prompts and tasks**
   - Baseline system prompt: `prompts/agent_base.txt`.
   - Intent system prompt template: `prompts/agent_with_intent.txt` (includes `{intent_block}`).
   - Tasks: `tasks/summarization_tasks.json` (4 tasks) and `tasks/planning_tasks.json` (4 tasks). Each task has `initial_intent`, optional `article`/`context`, and a list of `steps`. Summarization tasks are tied to public datasets: **XSum** (news, one-sentence summary), **PubMedQA** (biomedical abstract), **CUAD** (contract clauses), and **Amazon Product Reviews** (consumer pros/cons); each includes a `source_dataset` field and 7 refinement steps.

3. **Run baseline agent (no intent)**
   - For each task, the baseline agent:
     - Builds a conversation with the system prompt and user messages (first message includes article/context + first step).
     - At each step, calls the LLM once (deterministic: `do_sample=False`).
     - After step 0, computes IDS = `1 - cosine_similarity(embed(initial_output), embed(current_output))` using `sentence-transformers/all-MiniLM-L6-v2`.
     - Appends one JSONL record per step to `logs/baseline_runs.jsonl`.

4. **Run intent fusion agent**
   - For each task, the intent agent:
     - Creates an `Intent` object from `initial_intent` (goal, constraints, success_criteria, etc.).
     - At each step, builds the system prompt by injecting the **current Intent** (as a text block) into `agent_with_intent.txt`, so the model sees the intent on every turn.
     - Optionally checks proposed instruction vs. current intent (embedding similarity); if below threshold, keeps intent unchanged (and can log “conflict would trigger”); otherwise updates intent (e.g. `with_updated_goal`).
     - Generates the reply with the same LLM, then computes IDS the same way as baseline.
     - Appends one JSONL record per step to `logs/intent_runs.jsonl`.

5. **Interpret results**
   - `interpret_results()` in `run_experiment.py`:
     - Loads both JSONL logs, groups entries by `task_id`, and for each task computes mean and max IDS (over steps 1..N).
     - Compares baseline vs. intent per task (delta mean IDS, “winner” = whoever has lower mean IDS).
     - Writes:
       - `logs/task_comparison.csv` (per-task metrics),
       - `logs/summary_stats.csv` (overall and by task type),
       - `logs/ids_by_step.png`, `logs/ids_by_task_type.png`, `logs/ids_per_task.png`,
       - `logs/experiment_report.md` (summary tables and graph references).

So the “results” are: **same model, same tasks, same seed**; the only change is whether the agent gets an explicit, repeatedly injected Intent. Lower mean IDS for the intent agent on a task means better intent preservation on that task.

---

## Project Layout and Every File Explained

### Repository root

| File | Purpose |
|------|--------|
| **`README.md`** | This file: project overview, how results were accomplished, file-by-file explanation, results, and usage. |
| **`instructions.md`** | High-level experiment spec: goal, task types, baseline vs. intent fusion, IDS definition, logging format, and reproducibility requirements. Used as design doc, not executed. |
| **`requirements.txt`** | Pip dependencies: `transformers`, `torch`, `sentence-transformers`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`. |

---

### `intent_drift_experiment/` (experiment package)

#### Entry point and orchestration

| File | Purpose |
|------|--------|
| **`run_experiment.py`** | Main script. (1) Sets global seed. (2) Clears/creates `logs/` and truncates `baseline_runs.jsonl` and `intent_runs.jsonl`. (3) Loads prompts and both task JSONs. (4) Instantiates `BaselineAgent` and `IntentAgent`. (5) Runs all baseline tasks (writing to `baseline_runs.jsonl`), then all intent tasks (writing to `intent_runs.jsonl`). (6) Calls `interpret_results(logs_dir)` to generate CSVs, PNGs, and `experiment_report.md`. Defines helpers: `load_prompt()`, `load_tasks()`, `_load_jsonl()`, `_group_by_task_id()`, `_task_type()`, `_task_metrics()`, `_ids_by_step()`, and the full `interpret_results()` logic. `--interpret-only` skips running agents and only re-runs interpretation from existing logs. |

#### Intent data structure

| File | Purpose |
|------|--------|
| **`intent.py`** | Defines the **Intent** dataclass (single source of truth for goal): `intent_id`, `goal`, `constraints`, `success_criteria`, `assumptions`, `confidence`, `last_confirmed`, `version`. Provides `to_dict()`, `to_json()`, `to_prompt_block()` (compact text for the system prompt), `with_updated_goal()`, `with_confidence()`. Also provides `intent_factory()` to create an Intent from a goal (and optional fields) and `intent_from_dict()` for deserialization. The intent is sent to the agent every turn via the prompt. |

#### Model download (one-time)

| File | Purpose |
|------|--------|
| **`download_model.py`** | One-time download of: (1) **Qwen2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`) via `transformers` (tokenizer + causal LM). (2) **all-MiniLM-L6-v2** via `sentence_transformers` for IDS embeddings. Caches under Hugging Face cache (`~/.cache/huggingface/` or Windows equivalent). After this, you can set `HF_HUB_OFFLINE=1` to run offline. |

#### Prompts

| File | Purpose |
|------|--------|
| **`prompts/agent_base.txt`** | System prompt for the **baseline agent**: “You are a helpful AI assistant. Follow the user's instructions carefully.” No intent block. |
| **`prompts/agent_with_intent.txt`** | System prompt **template** for the **intent fusion agent**. Contains placeholder `{intent_block}`. Text instructs the model to follow the CURRENT INTENT exactly and to prioritize intent if the instruction conflicts. The intent block is injected every turn via `Intent.to_prompt_block()`. |

#### Tasks (fixed inputs)

| File | Purpose |
|------|--------|
| **`tasks/summarization_tasks.json`** | Array of 4 summarization tasks. Each object: `id`, `source_dataset` (e.g. "XSum (Extreme Summarization)", "PubMedQA (Biomedical Research Q&A)", "CUAD (Contract Understanding Atticus Dataset)", "Amazon Product Reviews (Electronics)"), `initial_intent`, `article` (source text), `steps` (list of 7 follow-up instructions). Tasks cover news (XSum), biomedical (PubMedQA), legal (CUAD), and consumer reviews (Amazon). Used for both agents. |
| **`tasks/planning_tasks.json`** | Array of 4 planning tasks. Each object: `id`, `initial_intent`, `context` (short description), `steps` (list of 7 follow-up instructions). Used for both agents. |
| **`tasks/tasks_metadata.json`** | Metadata only: for summarization—`count`, `source_datasets` (XSum, PubMedQA, CUAD, Amazon Reviews), `domains` (news, biomedical, legal, consumer), `complexity_levels`, `expected_drift_points`, `avg_steps`, `total_words_original`; for planning—count, domains, timeframes, avg_steps, complexity_levels. Not used by the run script; for documentation/reference. |

#### Agents

| File | Purpose |
|------|--------|
| **`agents/__init__.py`** | Exposes `BaselineAgent` and `IntentAgent`. |
| **`agents/baseline_agent.py`** | **BaselineAgent**: No intent. Uses Qwen2.5-7B-Instruct on CUDA, `do_sample=False`, fixed seed. For each task: builds conversation (first message = context + first step), runs steps via `run_baseline_step()`. At step 0, logs output and IDS=0; at steps 1..N, computes IDS via `metrics.ids.compute_ids(initial_output, current_output)` and logs. Handles long context by truncating or folding “previous response” to fit `MAX_MODEL_LENGTH`. Writes one JSONL line per step to the given log file. |
| **`agents/intent_agent.py`** | **IntentAgent**: Uses the same LLM and step runner as baseline but (1) creates an `Intent` from `initial_intent`, (2) at each step builds system prompt with `intent_prompt_with_object(intent)` (template + `intent.to_prompt_block()`), (3) optionally compares new instruction to current intent (embedding similarity); if below `INTENT_SIM_THRESHOLD`, keeps intent; else updates intent (e.g. `with_updated_goal`). Then runs one step, computes IDS the same way, and logs to JSONL (including intent fields). |

#### Metrics

| File | Purpose |
|------|--------|
| **`metrics/__init__.py`** | Re-exports `compute_ids`, `embedding`, `cosine_similarity` from `ids`. |
| **`metrics/ids.py`** | **Intent Drift Score**: Loads `sentence-transformers/all-MiniLM-L6-v2`. `embedding(text)` returns a vector. `cosine_similarity(a, b)` between two vectors. `compute_ids(initial_output, current_output)` = `1 - cosine_similarity(embed(initial), embed(current))`, clamped to [0, 1]. Empty text handling: both empty → 0; one empty → 1. |

#### Logs (outputs)

| File | Purpose |
|------|--------|
| **`logs/.gitkeep`** | Keeps `logs/` in version control when empty. |
| **`logs/baseline_runs.jsonl`** | One JSON object per line: `agent`, `task_id`, `step`, `prompt`, `output`, `ids` (0 at step 0, then float), `timestamp`. |
| **`logs/intent_runs.jsonl`** | One JSON object per line: `agent`, `task_id`, `step`, `intent` (dict), `intent_id`, `intent_version`, `intent_goal`, `prompt`, `output`, `ids`, optionally `conflict_would_trigger`. |
| **`logs/task_comparison.csv`** | Per-task comparison: `task_id`, `task_type`, `baseline_mean_ids`, `intent_mean_ids`, `baseline_max_ids`, `intent_max_ids`, `delta_mean`, `delta_max`, `winner` (baseline or intent). |
| **`logs/summary_stats.csv`** | Aggregate: rows for `overall`, `summarization`, `planning` with `baseline_mean_ids`, `intent_mean_ids`, `intent_wins`, `total_tasks`. |
| **`logs/experiment_report.md`** | Human-readable report: IDS explanation, summary table, task-level table, and references to the three PNGs. |
| **`logs/ids_by_step.png`** | Line plot: x = step, y = mean IDS across tasks; two series (baseline vs. intent). |
| **`logs/ids_by_task_type.png`** | Bar chart: mean IDS by task type (summarization, planning), baseline vs. intent. |
| **`logs/ids_per_task.png`** | Bar charts (per task type): per-task mean IDS, baseline vs. intent. |

---

## Results

Interpretation is generated by `interpret_results()` and written to `logs/`. Below is the kind of summary you get (exact numbers depend on the run).

### Summary statistics (example from `summary_stats.csv`)

| Scope         | Baseline mean IDS | Intent mean IDS | Intent wins | Total tasks |
|---------------|-------------------|-----------------|-------------|-------------|
| overall       | 0.8499            | 0.6786          | 4           | 8           |
| summarization | 0.8734            | 0.9073          | 1           | 4           |
| planning      | 0.8264            | 0.4498          | 3           | 4           |

- **Overall:** Intent fusion has **lower** mean IDS (0.6786 vs 0.8499), so on average it preserves intent better across all 8 tasks. Intent “wins” (lower mean IDS) in 4/8 tasks.
- **Summarization:** In the example run, baseline had lower mean IDS on 3/4 summarization tasks (intent wins 1/4). So on these multi-step summarization chains, intent fusion did not uniformly reduce drift.
- **Planning:** Intent fusion had much lower mean IDS (0.4498 vs 0.8264) and won 3/4 planning tasks, showing strong intent preservation for planning.

### Per-task comparison (example from `task_comparison.csv`)

- Each row is one task: `task_id`, `task_type`, baseline vs intent mean/max IDS, `delta_mean` (intent − baseline), and `winner`.
- Negative `delta_mean` ⇒ intent had lower mean IDS (intent wins). Positive ⇒ baseline wins.

### Graphs

- **`ids_by_step.png`:** IDS vs. step (mean across tasks). Lets you see if drift grows with steps and whether intent stays lower.
- **`ids_by_task_type.png`:** Mean IDS by task type (summarization vs. planning).
- **`ids_per_task.png`:** Mean IDS per task, split by type, baseline vs. intent.

So the **results** are: (1) overall, intent fusion reduces average drift (lower mean IDS) and wins in half the tasks; (2) the effect is strong on planning and mixed on summarization in the example run; (3) all numbers and plots are produced automatically from the two JSONL logs.

---

## Environment Setup

From the project root:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
# or: pip install transformers torch sentence-transformers numpy pandas scikit-learn matplotlib tqdm
```

### Download models (once)

Uses **Qwen2.5-7B-Instruct** (causal LM, CUDA) and **all-MiniLM-L6-v2** (IDS embeddings). Download once:

```bash
cd intent_drift_experiment
python download_model.py
```

Models are cached under:

- **Windows:** `C:\Users\<username>\.cache\huggingface\`
- **Mac/Linux:** `~/.cache/huggingface/`

### Run offline (optional)

After downloading, you can disable network for Hugging Face:

- **Windows (PowerShell):** `$env:HF_HUB_OFFLINE=1`
- **Windows (cmd):** `set HF_HUB_OFFLINE=1`
- **Mac/Linux:** `export HF_HUB_OFFLINE=1`

Then run the experiment with no API calls.

**Note:** The baseline and intent agents use **Qwen2.5-7B-Instruct** and expect CUDA. The IDS embedding model runs on CUDA if available, else CPU.

---

## Running the Experiment

```bash
cd intent_drift_experiment
python run_experiment.py
```

- Runs all baseline tasks, then all intent tasks.
- Writes `logs/baseline_runs.jsonl` and `logs/intent_runs.jsonl`.
- Then runs interpretation and writes CSVs, PNGs, and `experiment_report.md` under `logs/`.

To **only regenerate** tables and plots from existing logs (no model runs):

```bash
python run_experiment.py --interpret-only
```

---

## Interpretation and Outputs

After a full run (or after `--interpret-only` with existing logs):

- **Report:** `logs/experiment_report.md` — IDS definition, summary table, task-level table, and links to the PNGs.
- **Tables:** `logs/task_comparison.csv`, `logs/summary_stats.csv`.
- **Graphs:** `logs/ids_by_step.png`, `logs/ids_by_task_type.png`, `logs/ids_per_task.png`.

**How to read:**

- **IDS:** 0 = aligned with initial output (and thus initial intent); 1 = maximum drift; lower is better.
- **Winner per task:** Whichever agent has the lower mean IDS over steps 1..N.
- **Intent wins:** Number of tasks where intent fusion has lower mean IDS than baseline.

Together, the README, `instructions.md`, and this interpretation give a full, file-by-file and result-level explanation of how the results were accomplished and what each file does.
