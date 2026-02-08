# Intent Fusion Engine — Update Flowchart

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
        
        EMB_ORIG["embedding(intent.goal)"]
        EMB_PROP["embedding(instruction)"]
        CURRENT --> EMB_ORIG
        INSTRUCTION --> EMB_PROP
        
        SIM["sim = cosine_similarity(emb_orig, emb_proposed)"]
        EMB_ORIG --> SIM
        EMB_PROP --> SIM
        
        DECISION{"sim >= 0.30?"}
        SIM --> DECISION
        
        DECISION -->|Yes| ELABORATE["Elaborate"]
        DECISION -->|No| CONFLICT["Conflict"]
        
        ELABORATE --> WITH_ELAB["intent.with_elaboration(instruction)"]
        WITH_ELAB --> EXPANDED["Expanded Intent\ngoal + '. Additionally, the response must address: ' + instruction"]
        EXPANDED --> VERSION_UP["version += 1"]
        VERSION_UP --> NEXT_ELAB["next_intent = elaborated"]
        
        CONFLICT --> KEEP["next_intent = intent\n(unchanged)"]
        KEEP --> CONFLICT_NOTE["Add conflict_handling to prompt:\n'Follow format (bullets, table) if requested'"]
        CONFLICT_NOTE --> NEXT_CONF["next_intent = unchanged"]
    end

    subgraph Generation["Generation"]
        PROMPT["Build system prompt\nwith next_intent"]
        LLM["run_baseline_step()\nLLM generates response"]
        PROMPT --> LLM
    end

    INIT_INTENT --> CURRENT
    NEXT_ELAB --> PROMPT
    NEXT_CONF --> PROMPT
```

Note: `next_intent` becomes the current intent for the next step. The LLM generates the response but does not modify the intent.

## Summary

| Condition | Intent Update | Prompt Addendum |
|-----------|---------------|-----------------|
| **sim ≥ 0.30** | Elaborated: goal + "Additionally, the response must address: [instruction]" | None |
| **sim < 0.30** | Unchanged | Conflict note: follow requested format while staying aligned |

## with_elaboration()

```
expanded_goal = goal + ". Additionally, the response must address: " + instruction + "."
→ new Intent with version bumped
```

## Thresholds

- **LOW_THRESHOLD = 0.30**: Below this → treat as conflict (instruction diverges from intent)
- **HIGH_THRESHOLD = 0.75**: (Historical; currently sim ≥ LOW triggers elaboration)
