"""
Intent Drift Score (IDS): measures semantic divergence from the initial intent output.
IDS_t = 1 - cosine_similarity(embedding(initial_output), embedding(current_output))
IDS = 0 → perfectly aligned; IDS → 1 → maximum semantic drift.
"""
import numpy as np
from sentence_transformers import SentenceTransformer


# Default embedding model per instructions
IDS_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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


def compute_ids(initial_output: str, current_output: str) -> float:
    """
    Compute Intent Drift Score at a step.
    IDS_t = 1 - cosine_similarity(embedding(initial_output), embedding(current_output))
    """
    if not initial_output.strip() and not current_output.strip():
        return 0.0
    if not initial_output.strip() or not current_output.strip():
        return 1.0
    emb_init = embedding(initial_output)
    emb_curr = embedding(current_output)
    sim = cosine_similarity(emb_init, emb_curr)
    sim = max(-1.0, min(1.0, sim))
    ids = 1.0 - sim
    return float(max(0.0, min(1.0, ids)))  # IDS in [0, 1]: 0 = aligned, 1 = max drift
