# Intent Drift Score (IDS) — Calculation Flowchart

```mermaid
flowchart TB
    subgraph Context["Context (per task)"]
        S0["Step 0: LLM response"]
        ST["Step t: LLM response\n(t = 1, 2, ..., N)"]
    end

    subgraph Compute["compute_ids(initial_output, current_output)"]
        direction TB
        A["initial_output = Step 0 response"]
        B["current_output = Step t response"]
        
        A --> C{Both empty?}
        B --> C
        C -->|Yes| RETURN0["return 0.0"]
        C -->|No| D{One empty?}
        D -->|Yes| RETURN1["return 1.0"]
        D -->|No| E["emb_init = embedding(initial_output)"]
        E --> F["emb_curr = embedding(current_output)"]
        
        F --> G["sim = cosine_similarity(emb_init, emb_curr)"]
        G --> H["sim = clamp(sim, -1, 1)"]
        H --> I["IDS = 1 - sim"]
        I --> J["return clamp(IDS, 0, 1)"]
    end

    subgraph Embedding["embedding(text)"]
        K["SentenceTransformer\nall-MiniLM-L6-v2"]
        L["vector = model.encode(text)"]
        K --> L
    end

    subgraph Cosine["cosine_similarity(a, b)"]
        M["sim = dot(a, b) / (norm(a) × norm(b))"]
    end

    S0 --> A
    ST --> B
    E --> Embedding
    F --> Embedding
    G --> Cosine

    style S0 fill:#e3f2fd
    style ST fill:#e3f2fd
    style J fill:#c8e6c9
```

## Formula

```
IDS_t = 1 - cosine_similarity(embedding(initial_output), embedding(current_output))
```

- **IDS = 0** → perfectly aligned (same semantics as step 0)
- **IDS = 1** → maximum drift (orthogonal or opposite semantics)

## Aggregation (per task)

| Metric | Formula |
|--------|---------|
| **mean IDS** | mean(IDS_1, IDS_2, ..., IDS_N) — steps 1..N only |
| **max IDS** | max(IDS_1, ..., IDS_N) |

Step 0 has IDS = 0 (reference); drift is measured for steps 1 onward.
