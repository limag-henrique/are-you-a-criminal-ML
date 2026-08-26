"""Identity-group reconstruction from cosine-similarity connected components."""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


def cosine_similarity_groups(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Return component identifiers for edges whose cosine similarity meets threshold."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    values = np.asarray(embeddings, dtype=float)
    graph = NearestNeighbors(metric="cosine").fit(values).radius_neighbors_graph(
        values, radius=1.0 - threshold + 1e-12, mode="connectivity"
    )
    graph = graph.maximum(graph.T)
    _, labels = connected_components(graph, directed=False)
    return labels.astype(int)

