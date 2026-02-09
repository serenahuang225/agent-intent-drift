# Intent Drift Experiment — Data Flow

```mermaid
flowchart TB
    subgraph INPUTS["📥 Inputs"]
        TASKS["tasks/summarization_tasks.json\nplanning_tasks.json"]
        PROMPTS["prompts/agent_base.txt\nagent_with_intent.txt"]
        MODELS["MODELS list\n(Llama, Gemma, Mistral)"]
    end

    subgraph ENTRY["🚀 Entry Points"]
        RUN["run_experiment.py"]
        DEBUG["debug_intent.py"]
    end

    subgraph RUN_FLOW["run_experiment.py Flow"]
        direction TB
        RUN --> LOAD_PROMPT["load_prompt()"]
        RUN --> LOAD_TASKS["load_tasks()"]
        LOAD_PROMPT --> PROMPTS
        LOAD_TASKS --> TASKS
        
        RUN --> BASELINE_LOOP["For each model × task × run:"]
        BASELINE_LOOP --> BASELINE_AGENT["BaselineAgent"]
        BASELINE_AGENT --> BASELINE_STEP["run_baseline_step()"]
        BASELINE_STEP --> LLM_B["LLM (transformers)"]
        BASELINE_STEP --> BASELINE_LOG["Append to baseline_runs_{slug}.jsonl"]
        
        RUN --> INTENT_LOOP["For each model × task × run:"]
        INTENT_LOOP --> INTENT_AGENT["IntentAgent"]
        INTENT_AGENT --> INTENT_FACTORY["intent_factory()"]
        INTENT_FACTORY --> INTENT_PY["intent.py\nderive_constraints_from_goal()"]
        INTENT_AGENT --> INTENT_STEP["run_intent_step()"]
        INTENT_STEP --> SIM["metrics/ids.py\nembedding() + cosine_similarity()"]
        SIM --> ELAB_OR_CONFLICT{"sim >= 0.30?"}
        ELAB_OR_CONFLICT -->|Yes| ELAB["with_elaboration()"]
        ELAB_OR_CONFLICT -->|No| CONFLICT["Keep intent + conflict_handling"]
        ELAB --> BASELINE_STEP
        CONFLICT --> BASELINE_STEP
        INTENT_STEP --> COMPUTE_IDS["compute_ids()\n1 - cosine_sim(output, current_goal)"]
        INTENT_AGENT --> INTENT_LOG["Append to intent_runs_{slug}.jsonl"]
    end

    subgraph DEBUG_FLOW["debug_intent.py Flow"]
        DEBUG --> DEBUG_LOAD["load_tasks()"]
        DEBUG_LOAD --> TASKS
        DEBUG --> INTENT_FACTORY
        DEBUG --> INTENT_AGENT
        DEBUG --> DEBUG_PRINT["Print similarity, conflict flags,\nstep outputs, analysis"]
    end

    subgraph LOGS["📄 Log Outputs (logs/)"]
        BASELINE_JSONL["baseline_runs_{slug}.jsonl"]
        INTENT_JSONL["intent_runs_{slug}.jsonl"]
    end

    subgraph INTERPRET["📊 Interpretation (interpret_all_models)"]
        direction TB
        LOAD_JSONL["_load_jsonl()"]
        LOAD_JSONL --> BASELINE_JSONL
        LOAD_JSONL --> INTENT_JSONL
        
        GROUP["_group_by_task_id()"]
        LOAD_JSONL --> GROUP
        
        METRICS["_task_metrics()\nmean IDS, max IDS (steps 1..N)"]
        GROUP --> METRICS
        
        METRICS --> TASK_ROWS["task_rows\n(baseline_mean, intent_mean, winner)"]
        
        TASK_ROWS --> SIG["_paired_ids_significance()\nt-test, Wilcoxon, Cohen's d"]
        
        TASK_ROWS --> PER_MODEL["Per model: interpret_results()"]
        PER_MODEL --> PER_MODEL_CSV["{slug}/task_comparison.csv\n{slug}/summary_stats.csv\n{slug}/significance_stats.csv"]
        PER_MODEL --> PER_MODEL_PNG["{slug}/ids_by_step.png\n{slug}/ids_by_task_type.png\n{slug}/ids_per_task.png"]
        PER_MODEL --> PER_MODEL_MD["{slug}/experiment_report.md"]
        
        TASK_ROWS --> CROSS_VIZ["_write_cross_model_visualizations()"]
        CROSS_VIZ --> CROSS_PNG["ids_by_model.png\nids_by_task_type_by_model.png\nids_by_task_type_all_models.png\nids_per_task_avg_models.png\nids_by_step_all_models.png\nids_by_step_avg_models.png"]
        CROSS_VIZ --> CROSS_CSV["cross_model_summary.csv"]
        CROSS_VIZ --> OVERALL_REPORT["_write_overall_experiment_report()"]
        OVERALL_REPORT --> OVERALL_MD["experiment_report.md\n(overall p-value)"]
    end

    subgraph DEPENDENCIES["Dependencies"]
        INTENT_PY
        METRICS_MOD["metrics/ids.py\n(embedding, compute_ids, compute_goal_shift)"]
        BASELINE_MOD["agents/baseline_agent.py"]
        INTENT_MOD["agents/intent_agent.py"]
    end

    RUN --> INTERPRET
    BASELINE_LOG --> BASELINE_JSONL
    INTENT_LOG --> INTENT_JSONL
```

## Data Flow Summary

### 1. Input → Execution

| Source | Data | Consumed By |
|--------|------|-------------|
| `tasks/*.json` | `initial_intent`, `steps`, `article`/`context` | run_experiment, debug_intent |
| `prompts/*.txt` | System prompts (baseline vs intent-aware) | BaselineAgent, IntentAgent |
| MODELS | HuggingFace model names | Both agents |

### 2. Agent Execution

**BaselineAgent**: Uses `agent_base.txt` → builds conversation → LLM generates → no IDS computed during run (IDS computed later in interpret).

**IntentAgent**: Uses `agent_with_intent.txt` + Intent block → at each step:
- Computes `sim = cosine_similarity(goal_emb, instruction_emb)`
- If sim ≥ 0.30: elaborates intent
- If sim < 0.30: keeps intent + adds conflict_handling note
- Calls `run_baseline_step()` (shared with BaselineAgent)
- Computes `IDS_t = 1 - cosine_similarity(initial_output, current_output)` per step

### 3. Log Format (JSONL)

Each line: `{task_id, step, output, ids, ...}`. Grouped by `task_id` during interpretation.

### 4. Interpretation Pipeline

1. **Load** baseline + intent JSONL → group by task_id
2. **Metrics** per task: mean IDS (steps 1..N), max IDS
3. **Compare**: baseline_mean vs intent_mean → winner
4. **Per-model**: CSV, PNG, report in `logs/{slug}/`
5. **Cross-model**: aggregate PNGs, `cross_model_summary.csv`
6. **Overall**: pool all (model, task) pairs → paired t-test → `experiment_report.md`

### 5. Output Files

| Location | Files |
|----------|-------|
| `logs/` | `experiment_report.md`, `cross_model_summary.csv`, `ids_*.png` |
| `logs/{model-slug}/` | `task_comparison.csv`, `summary_stats.csv`, `significance_stats.csv`, `experiment_report.md`, `ids_*.png` |
