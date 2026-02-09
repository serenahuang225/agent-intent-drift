# Intent Fusion Engine — Update Flowchart

Inverted logic: **sim > 0.8 → keep**; **sim < 0.6 → replace goal**; **0.6 ≤ sim ≤ 0.8 → replace + log high_drift_risk**. No blocking; current intent is always sent to the model.

```mermaid
flowchart TB
    subgraph Init["Step 0 (Initialization)"]
        USER_INTENT["User's initial_intent\ne.g. 'Summarize in one sentence'"]
        FACTORY["intent_factory()"]
        INIT_INTENT["Intent created\n(goal, constraints, success_criteria)"]
        USER_INTENT --> FACTORY --> INIT_INTENT
    end

    subgraph Step["Steps 1..N (run_intent_step)"]
        direction TB
        CURRENT["Current Intent"]
        INSTRUCTION["New instruction\n(user_message)"]

        EMB_GOAL["embedding(intent.goal)"]
        EMB_INST["embedding(instruction)"]
        CURRENT --> EMB_GOAL
        INSTRUCTION --> EMB_INST

        SIM["sim = cosine_similarity(emb_goal, emb_inst)"]
        EMB_GOAL --> SIM
        EMB_INST --> SIM

        DECISION{"sim > 0.8?"}
        SIM --> DECISION

        DECISION -->|Yes| REFINE["Refinement"]
        DECISION -->|No| DEC2{"sim < 0.6?"}

        REFINE --> KEEP["next_intent = intent\n(unchanged)"]
        KEEP --> UPDATE_TYPE_REF["update_type = 'refinement'"]

        DEC2 -->|Yes| MAJOR["Major shift"]
        DEC2 -->|No| AMBIG["Ambiguous"]

        MAJOR --> REPLACE_MAJ["intent.with_goal_replacement(instruction)\nreason: 'major_shift'"]
        REPLACE_MAJ --> UPDATE_TYPE_MAJ["update_type = 'major_shift'"]

        AMBIG --> REPLACE_RISK["intent.with_goal_replacement(instruction)\nreason: 'high_drift_risk'"]
        REPLACE_RISK --> UPDATE_TYPE_RISK["update_type = 'high_drift_risk'"]
    end

    subgraph Generation["Generation"]
        PROMPT["Build system prompt\nwith next_intent (no conflict note)"]
        LLM["run_baseline_step()\nLLM generates response"]
        PROMPT --> LLM
    end

    INIT_INTENT --> CURRENT
    KEEP --> PROMPT
    REPLACE_MAJ --> PROMPT
    REPLACE_RISK --> PROMPT
```

Note: `next_intent` becomes the current intent for the next step. The LLM always receives the current intent; there is no conflict block or refusal path.

## Summary

| Condition       | Intent update                         | update_type       |
|----------------|---------------------------------------|-------------------|
| **sim > 0.8**  | Keep (no change)                      | `refinement`      |
| **sim < 0.6**  | Replace goal via `with_goal_replacement(instruction)`, append to update_history with reason `major_shift` | `major_shift`     |
| **0.6 ≤ sim ≤ 0.8** | Replace goal via `with_goal_replacement(instruction)`, append with reason `high_drift_risk` | `high_drift_risk` |

## with_goal_replacement(new_goal)

- Replaces `goal` with `new_goal`.
- Re-derives `constraints` and `success_criteria` via `derive_constraints_from_goal(new_goal)`.
- Bumps `version` by 1.
- Optionally appends an entry to `update_history` (e.g. `{ old_goal, new_goal, reason }`).

## Thresholds

- **HIGH_THRESHOLD = 0.8**: Above → refinement; keep intent.
- **LOW_THRESHOLD = 0.6**: Below → major shift; replace goal.
- **Between 0.6 and 0.8**: Update goal and log as high_drift_risk.

## intent_replay

The same logic is replayed over the instruction sequence (without running the LLM) to obtain **inferred intent per step** for IDS and goal_shift computation during interpretation.
