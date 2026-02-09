"""
IDS (Intent Drift Score): per-step divergence of output from current inferred goal.
Goal shift: task-level cumulative user-led change = 1 - sim(initial_goal, final_goal).

Tuned for semantic/real drift (meaning and task adherence) rather than literal match:
- Stronger semantic embedding model (all-mpnet-base-v2 by default).
- Optional semantic scoring: score by best-matching part of output to goal (max over sentences),
  so paraphrases and verbose-but-on-topic answers are not over-penalized.

- compute_ids(output, current_goal): Intent Drift Score; 1 - similarity; lower = more aligned.
- compute_goal_shift(initial_goal, final_goal): 1 - sim(initial_goal, final_goal).
Both in [0, 1]; lower is better (0 = aligned).
"""
import re
import numpy as np
from sentence_transformers import SentenceTransformer


# More semantic model: better paraphrase and meaning similarity (less literal)
IDS_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[IDS] Loading embedding model: {} ...".format(IDS_EMBEDDING_MODEL))
        _model = SentenceTransformer(IDS_EMBEDDING_MODEL, device=device)
        print("[IDS] Embedding model ready on {}.".format(device))
    return _model


def embedding(text: str) -> np.ndarray:
    """Get embedding for a single text."""
    return _get_model().encode(text, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _sentences(text: str) -> list:
    """Split text into sentences (simple; for semantic IDS chunking)."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence boundaries, keep non-empty
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _goal_gist(goal: str, max_chars: int = 200) -> str:
    """Short gist of the goal for more forgiving semantic comparison (reduces IDS when output addresses the main ask)."""
    g = goal.strip()
    if len(g) <= max_chars:
        return g
    # Prefer first sentence(s) up to max_chars
    first = re.split(r"(?<=[.!?])\s+", g, maxsplit=1)[0]
    if len(first) <= max_chars:
        return first
    return g[:max_chars].rsplit(maxsplit=1)[0] or g[:max_chars]


def compute_ids(
    output: str,
    current_goal: str,
    semantic: bool = True,
    use_goal_gist: bool = True,
) -> float:
    """
    Intent Drift Score: how much the output diverges from the current goal (per-step).

    When semantic=True (default): measures semantic/real drift, not literal match.
    - Embedding model is chosen for meaning similarity (all-mpnet-base-v2).
    - Similarity is taken as the *max* over output sentences vs goal: if any part of
      the output semantically addresses the goal, drift is low.
    - When use_goal_gist=True (default): if the goal is long, we also compare to a
      short gist (first ~200 chars) and use the *max* similarity. Reduces IDS when
      the output addresses the main ask but the full instruction has extra clauses.

    When semantic=False: whole-output similarity (original behavior).

    Returns value in [0, 1]; lower is better (0 = aligned).
    """
    if not output.strip() and not current_goal.strip():
        return 0.0
    if not output.strip() or not current_goal.strip():
        return 1.0
    # Optionally compare to gist as well (more forgiving for long instructions)
    goals_to_compare = [current_goal.strip()]
    if use_goal_gist and len(current_goal.strip()) > 200:
        goals_to_compare.append(_goal_gist(current_goal))
    emb_goals = [embedding(g) for g in goals_to_compare]
    if semantic:
        sentences = _sentences(output)
        if not sentences:
            emb_out = embedding(output)
            sim = max(cosine_similarity(emb_out, eg) for eg in emb_goals)
        else:
            embs = [embedding(s) for s in sentences]
            sim = max(
                cosine_similarity(e, eg)
                for e in embs
                for eg in emb_goals
            )
    else:
        emb_out = embedding(output)
        sim = max(cosine_similarity(emb_out, eg) for eg in emb_goals)
    sim = max(-1.0, min(1.0, sim))
    ids = 1.0 - sim
    return float(max(0.0, min(1.0, ids)))


def compute_goal_shift(initial_goal: str, final_goal: str) -> float:
    """
    Task-level cumulative goal shift: 1 - sim(embedding(initial_goal), embedding(final_goal)).
    Measures user-led change from initial to final intent. Lower = less cumulative shift.
    """
    if not initial_goal.strip() and not final_goal.strip():
        return 0.0
    if not initial_goal.strip() or not final_goal.strip():
        return 1.0
    emb_init = embedding(initial_goal)
    emb_final = embedding(final_goal)
    sim = cosine_similarity(emb_init, emb_final)
    sim = max(-1.0, min(1.0, sim))
    shift = 1.0 - sim
    return float(max(0.0, min(1.0, shift)))
