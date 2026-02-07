"""
Download models once. After this, you can run the experiment offline.

- Causal LMs: Llama-3.1-8B, Gemma-2-2B-IT, Mistral-7B-Instruct (see CAUSAL_MODELS)
- Embeddings: all-MiniLM-L6-v2 (for Intent Drift Score)

Models are cached by Hugging Face in:
  Windows: C:\\Users\\<username>\\.cache\\huggingface\\
  Mac/Linux: ~/.cache/huggingface/

Then set HF_HUB_OFFLINE=1 to run fully offline (no API calls).
"""

import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Align with run_experiment.MODELS so one script caches all
CAUSAL_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


def _get_hf_token():
    """Use HF_TOKEN env or cached login so gated models (e.g. Llama) can be downloaded."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _model_cached(repo_id: str) -> bool:
    """Return True if the repo is already in the Hugging Face cache (no network, no full load)."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def download_causal_model(model_name: str):
    """Download a single causal LM and tokenizer. Skips if already in cache. Gemma/Llama may require HF token."""
    if _model_cached(model_name):
        print(f"  {model_name} already in cache, skipping download.")
        return None, None

    token = _get_hf_token()
    if "llama" in model_name.lower() or "meta-llama" in model_name:
        if not token:
            print("Hint: Llama is gated. Set HF_TOKEN or run: huggingface-cli login")
    if "gemma" in model_name.lower():
        if not token:
            print("Hint: Gemma may require accepting the license on Hugging Face and HF_TOKEN.")
    print(f"Downloading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, token=token
    )
    print(f"  {model_name} downloaded successfully.")
    return model, tokenizer


def download_embedding_model():
    """Download embedding model for IDS. Skips if already in cache."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if _model_cached(model_name):
        print(f"  {model_name} already in cache, skipping download.")
        return None

    print(f"Downloading {model_name}...")
    embedder = SentenceTransformer(model_name)
    print("Embedding model ready!")
    return embedder

if __name__ == "__main__":
    print("=" * 60)
    print("Downloading models (run once). Cached for offline use.")
    print("=" * 60)
    for i, model_name in enumerate(CAUSAL_MODELS, start=1):
        print("\n[Step {}/{}] Causal LM: {}".format(i, len(CAUSAL_MODELS) + 1, model_name))
        download_causal_model(model_name)
    print("\n[Step {}/{}] Embedding model (for Intent Drift Score)".format(len(CAUSAL_MODELS) + 1, len(CAUSAL_MODELS) + 1))
    download_embedding_model()
    print("\n" + "=" * 60)
    print("Done. You can now run the experiment offline with: set HF_HUB_OFFLINE=1")
    print("=" * 60)
