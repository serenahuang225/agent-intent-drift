# IDS and Goal Shift — Calculation Flowchart

## Primary: IDS (Intent Drift Score)

Per-step divergence of **model output** from the **current inferred goal** (from intent_replay). This is the **Intent Drift Score (IDS)**.

```mermaid
flowchart TB
    subgraph Context["Context (per step t)"]
        OUTPUT["Step t: LLM output"]
        GOAL["Inferred goal at step t\n(from intent_replay)"]
    end

    subgraph ComputeIDS["compute_ids(output, current_goal)"]
        direction TB
        A["output = Step t response"]
        B["current_goal = inferred_goal_t"]
        A --> C{Both empty?}
        B --> C
        C -->|Yes| RETURN0["return 0.0"]
        C -->|No| D{One empty?}
        D -->|Yes| RETURN1["return 1.0"]
        D -->|No| E["emb_out = embedding(output)"]
        E --> F["emb_goal = embedding(current_goal)"]
        F --> G["sim = cosine_similarity(emb_out, emb_goal)"]
        G --> H["sim = clamp(sim, -1, 1)"]
        H --> I["IDS = 1 - sim"]
        I --> J["return clamp(IDS, 0, 1)"]
    end

    OUTPUT --> A
    GOAL --> B
    E --> Embedding
    F --> Embedding
    G --> Cosine

    subgraph Embedding["embedding(text)"]
        K["SentenceTransformer\nall-MiniLM-L6-v2"]
        L["vector = model.encode(text)"]
        K --> L
    end

    subgraph Cosine["cosine_similarity(a, b)"]
        M["sim = dot(a, b) / (norm(a) × norm(b))"]
    end

    style OUTPUT fill:#e3f2fd
    style GOAL fill:#fff3e0
    style J fill:#c8e6c9
```

**Formula:** `IDS_t = 1 - cosine_similarity(embedding(output_t), embedding(inferred_goal_t))`

- **IDS = 0** → output aligned with current inferred goal  
- **IDS = 1** → maximum drift from current goal  

Inferred goal at each step comes from **intent_replay(initial_intent, steps)** (same thresholds as the intent agent).

---

## Secondary: Goal shift (task-level cumulative goal shift)

Task-level measure of **user-led** change from initial to final intent.

```mermaid
flowchart TB
    subgraph Input["Input (per task)"]
        INIT["initial_intent\n(task goal)"]
        FINAL["final inferred goal\n(inferred_per_step[-1]['goal'])"]
    end

    subgraph ComputeGoalShift["compute_goal_shift(initial_goal, final_goal)"]
        direction TB
        A["initial_goal = task initial_intent"]
        B["final_goal = inferred intent at last step"]
        A --> C{Both empty?}
        B --> C
        C -->|Yes| R0["return 0.0"]
        C -->|No| D{One empty?}
        D -->|Yes| R1["return 1.0"]
        D -->|No| E["emb_init = embedding(initial_goal)"]
        E --> F["emb_final = embedding(final_goal)"]
        F --> G["sim = cosine_similarity(emb_init, emb_final)"]
        G --> H["goal_shift = 1 - sim"]
        H --> J["return clamp(goal_shift, 0, 1)"]
    end

    INIT --> A
    FINAL --> B
    E --> Embedding
    F --> Embedding
    G --> Cosine

    subgraph Embedding["embedding(text)"]
        K["SentenceTransformer\nall-MiniLM-L6-v2"]
        K --> L["vector = model.encode(text)"]
    end

    subgraph Cosine["cosine_similarity(a, b)"]
        M["sim = dot(a, b) / (norm(a) × norm(b))"]
    end

    style INIT fill:#e3f2fd
    style FINAL fill:#fff3e0
    style J fill:#c8e6c9
```

**Formula:** `goal_shift = 1 - cosine_similarity(embedding(initial_goal), embedding(final_goal))`

- **goal_shift = 0** → no cumulative goal shift (final intent same as initial)  
- **goal_shift = 1** → maximum cumulative shift  

Same for both baseline and intent agents (task-defined initial and final inferred goal).

---

## Aggregation (per task)

| Metric   | Formula |
|----------|---------|
| **mean IDS** | mean(IDS_1, IDS_2, ..., IDS_N) — steps 1..N only |
| **max IDS**  | max(IDS_1, ..., IDS_N) |
| **goal_shift** | One value per task (initial vs final inferred goal) |

Step 0 can have IDS computed vs inferred goal at step 0; reporting typically uses steps 1..N for mean/max IDS.
