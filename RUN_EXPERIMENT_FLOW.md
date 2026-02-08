# run_experiment.py — Data Flow

```mermaid

flowchart TB
    subgraph MAIN["main()"]
        M1["[1/5] Models, Seeds"]
        M2["[2/5] Prepare logs dir Clear prev runs if needed"]
        M3["[3/5] Load prompts & tasks"]
        M4["[4/5] Run agents (per model)"]
        M5["[5/5] interpret_all_models()"]
        M1 --> M2 --> M3 --> M4 --> M5
    end

    subgraph LOAD["Load inputs"]
        PROMPT["load_prompt('agent_base.txt')"]
        TASKS["load_tasks() summarization_tasks.json planning_tasks.json"]
    end

    subgraph LOOP["For each model"]
        direction TB
        BASELINE_PHASE["--- Baseline agent ---"]
        INTENT_PHASE["--- Intent fusion agent ---"]
        
        BASELINE_PHASE --> BL_LOOP["For each task × run: BaselineAgent.run_and_log()"]
        BL_LOOP --> BL_APPEND["Append → baseline_runs_{slug}.jsonl"]
        
        BASELINE_PHASE --> FREE_GPU["Free model, gc, empty_cache"]
        FREE_GPU --> INTENT_PHASE
        
        INTENT_PHASE --> INT_LOOP["For each task × run: IntentAgent.run_and_log()"]
        INT_LOOP --> INT_APPEND["Save Data by Appending → intent_runs_{model_slug}.jsonl"]
    end


    M3 --> PROMPT
    M3 --> TASKS
    M4 --> LOOP
    LOAD --> LOOP
    LOOP --> INTERPRET
```

## Step-by-step

| Step | Action |
|------|--------|
| 1 | Models list + seeds (42, 43, 44) |
| 2 | Create `logs/`, clear previous run logs for models being run |
| 3 | Load `agent_base.txt`, `summarization_tasks.json`, `planning_tasks.json` |
| 4 | For each model (if logs missing): run BaselineAgent then IntentAgent, append to JSONL |
| 5 | `interpret_all_models()`: load JSONL → per-model interpretation → cross-model viz → overall report |

## Outputs

| Path | Content |
|------|---------|
| `logs/baseline_runs_{slug}.jsonl` | Per-step baseline logs (task_id, step, output, ids) |
| `logs/intent_runs_{slug}.jsonl` | Per-step intent logs |
| `logs/{slug}/` | Per-model CSVs, PNGs, experiment_report.md |
| `logs/` | Cross-model PNGs, cross_model_summary.csv, experiment_report.md |
