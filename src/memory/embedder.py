"""
fastembed BAAI/bge-small-en-v1.5 — local ONNX, ~5-15ms per call, 384-dim L2-normalised.
Model (~67MB) is downloaded once on first use and cached in ~/.cache/fastembed/.
dot(a, b) == cosine_similarity(a, b) because outputs are L2-normalised.
"""
import numpy as np
from fastembed import TextEmbedding

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(_MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """Return a 384-dim L2-normalised float32 vector for one text."""
    results = list(_get_model().embed([text]))
    return np.array(results[0], dtype="float32")


def embed_batch(texts: list[str]) -> np.ndarray:
    """Return a (N, 384) float32 matrix. Rows match input order."""
    results = list(_get_model().embed(texts))
    return np.array(results, dtype="float32")
