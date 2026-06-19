"""
Review embedding and clustering engine (Phase 4).

Pipeline:
  1. embed_reviews  — sentence-transformers → numpy array of embeddings
  2. cluster_reviews — UMAP dimensionality reduction → HDBSCAN density clustering

Data model produced:
  RawCluster: { cluster_id, review_indices, avg_rating, review_count }

Architecture references:
  - §4   — Analysis engine
  - §10  — Error handling: 0-cluster abort
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("pulse.clustering")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawCluster:
    """
    A cluster of reviews produced by HDBSCAN, before LLM summarization.

    Attributes:
        cluster_id:     Integer label assigned by HDBSCAN (0-indexed).
        review_indices: Indices into the original reviews list.
        avg_rating:     Mean star rating of reviews in this cluster.
        review_count:   Number of reviews in the cluster.
    """
    cluster_id: int
    review_indices: list[int]
    avg_rating: float
    review_count: int


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_reviews(reviews: list[dict], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate sentence embeddings for a list of review dicts.

    Uses ``sentence-transformers`` to encode the ``text`` field of each review.
    The resulting array has shape ``(len(reviews), embedding_dim)`` where
    ``embedding_dim`` depends on the chosen model:
      - BAAI/bge-small-en-v1.5  → 384 dims (primary, English-optimised)
      - all-MiniLM-L6-v2        → 384 dims (free fallback, most widely cached)

    If the configured model fails to load (download error, model unavailable),
    the function automatically retries with ``all-MiniLM-L6-v2`` before raising.
    Both models are 100% local and free — no API key required.

    Args:
        reviews:    List of review dicts, each with a ``text`` field.
        model_name: sentence-transformers model identifier.
                    Default: ``all-MiniLM-L6-v2`` (fast, good quality, widely cached).

    Returns:
        numpy array of shape (N, D) — N reviews × D embedding dimensions.

    Raises:
        ImportError: If sentence-transformers is not installed.
        ValueError:  If the reviews list is empty.
        RuntimeError: If both the primary and fallback models fail to load.
    """
    if not reviews:
        raise ValueError("Cannot embed an empty reviews list.")

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install with: pip install sentence-transformers"
        ) from exc

    # Free fallback: all-MiniLM-L6-v2 is the most widely cached ST model
    _FALLBACK_MODEL = "all-MiniLM-L6-v2"

    texts = [r.get("text", "") for r in reviews]

    def _load_and_encode(model_id: str) -> np.ndarray:
        logger.info(
            "Generating embeddings for %d reviews using model '%s'...",
            len(texts), model_id,
        )
        model = SentenceTransformer(model_id)
        result = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        logger.info(
            "Embeddings generated: shape=%s, dtype=%s", result.shape, result.dtype
        )
        return result

    # --- Try primary model, fall back to all-MiniLM-L6-v2 if it fails ---
    try:
        return _load_and_encode(model_name)
    except Exception as primary_exc:
        if model_name == _FALLBACK_MODEL:
            # Already using fallback — nothing left to try
            raise RuntimeError(
                f"Embedding model '{model_name}' failed to load: {primary_exc}"
            ) from primary_exc

        logger.warning(
            "Primary embedding model '%s' failed (%s). "
            "Retrying with free fallback model '%s'...",
            model_name, primary_exc, _FALLBACK_MODEL,
        )
        try:
            return _load_and_encode(_FALLBACK_MODEL)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Both primary model '{model_name}' and fallback '{_FALLBACK_MODEL}' "
                f"failed to load. Primary error: {primary_exc}. "
                f"Fallback error: {fallback_exc}"
            ) from fallback_exc



# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_reviews(
    embeddings: np.ndarray,
    reviews: list[dict],
    config: dict,
) -> list[RawCluster]:
    """
    Cluster review embeddings using UMAP dimensionality reduction + HDBSCAN.

    Steps:
      1. Reduce embedding dimensionality with UMAP.
      2. Apply HDBSCAN density clustering.
      3. Filter out noise (label == -1).
      4. Sort clusters by size (descending).
      5. Cap at ``config["clustering"]["max_themes"]``.

    Args:
        embeddings: numpy array of shape (N, D) from ``embed_reviews``.
        reviews:    Original reviews list (used to extract ratings).
        config:     Full config dict (reads ``clustering`` section).

    Returns:
        List of ``RawCluster`` objects, sorted largest-first, capped at max_themes.

    Raises:
        RuntimeError: If 0 clusters are produced (triggers pipeline abort).
        ImportError:  If umap-learn or hdbscan is not installed.
    """
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for clustering. "
            "Install with: pip install umap-learn"
        ) from exc

    try:
        import hdbscan as hdbscan_lib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "hdbscan is required for clustering. "
            "Install with: pip install hdbscan"
        ) from exc

    clustering_cfg = config.get("clustering", {})
    n_neighbors: int = clustering_cfg.get("umap_n_neighbors", 15)
    n_components: int = clustering_cfg.get("umap_n_components", 5)
    min_cluster_size: int = clustering_cfg.get("hdbscan_min_cluster_size", 5)
    max_themes: int = clustering_cfg.get("max_themes", 8)

    n_reviews = embeddings.shape[0]
    logger.info(
        "Clustering %d reviews: UMAP(n_neighbors=%d, n_components=%d) → "
        "HDBSCAN(min_cluster_size=%d)",
        n_reviews, n_neighbors, n_components, min_cluster_size,
    )

    # Clamp n_neighbors to avoid UMAP errors on small datasets
    effective_n_neighbors = min(n_neighbors, max(2, n_reviews - 1))
    if effective_n_neighbors != n_neighbors:
        logger.warning(
            "n_neighbors clamped from %d to %d (too few reviews).",
            n_neighbors, effective_n_neighbors,
        )

    # --- Step 1: UMAP reduction ---
    reducer = umap.UMAP(
        n_neighbors=effective_n_neighbors,
        n_components=min(n_components, n_reviews - 1),
        random_state=42,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    logger.info("UMAP reduction complete: shape=%s", reduced.shape)

    # --- Step 2: HDBSCAN clustering ---
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min(min_cluster_size, max(2, n_reviews // 4)),
        prediction_data=True,
    )
    labels = clusterer.fit_predict(reduced)

    unique_labels = set(labels) - {-1}  # -1 is noise
    logger.info(
        "HDBSCAN found %d clusters (%d noise points).",
        len(unique_labels),
        int((labels == -1).sum()),
    )

    # --- Step 3: Abort on 0 clusters (architecture §10) ---
    if not unique_labels:
        raise RuntimeError(
            "HDBSCAN produced 0 clusters from the review embeddings. "
            "The reviews may be too diverse or the dataset too small. "
            "Aborting run — no meaningful themes to report."
        )

    # --- Step 4: Build RawCluster objects ---
    raw_clusters: list[RawCluster] = []
    for label in unique_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        ratings = [reviews[i].get("rating", 0) for i in indices]
        avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else 0.0
        raw_clusters.append(RawCluster(
            cluster_id=int(label),
            review_indices=indices,
            avg_rating=avg_rating,
            review_count=len(indices),
        ))

    # --- Step 5: Sort by size descending, cap at max_themes ---
    raw_clusters.sort(key=lambda c: c.review_count, reverse=True)
    raw_clusters = raw_clusters[:max_themes]

    logger.info(
        "Returning %d clusters (max_themes=%d). Sizes: %s",
        len(raw_clusters),
        max_themes,
        [c.review_count for c in raw_clusters],
    )
    return raw_clusters
