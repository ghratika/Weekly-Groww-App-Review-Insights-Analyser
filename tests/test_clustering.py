"""
Unit tests for src/agent/clustering.py (Phase 4).

Tests cover:
  - embed_reviews: output shape and dtype
  - cluster_reviews: expected cluster count from synthetic data
  - cluster_reviews: 0-cluster edge case triggers RuntimeError (abort)
  - cluster_reviews: honours max_themes cap
  - embed_reviews: empty list raises ValueError
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.agent.clustering import RawCluster, cluster_reviews, embed_reviews


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_reviews(n: int, base_rating: int = 3) -> list[dict]:
    """Create n minimal review dicts with synthetic text."""
    return [
        {
            "review_id": f"r{i}",
            "text": f"Review number {i}: this is synthetic test content.",
            "rating": (i % 5) + 1,
        }
        for i in range(n)
    ]


def _fake_embeddings(n: int, dim: int = 384) -> np.ndarray:
    """Return a deterministic random embedding matrix."""
    rng = np.random.RandomState(42)
    return rng.rand(n, dim).astype(np.float32)


def _cluster_config(**overrides) -> dict:
    """Build a minimal clustering config dict."""
    cfg = {
        "clustering": {
            "umap_n_neighbors": 5,
            "umap_n_components": 2,
            "hdbscan_min_cluster_size": 2,
            "max_themes": 8,
        }
    }
    cfg["clustering"].update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# embed_reviews tests
# ---------------------------------------------------------------------------

class TestEmbedReviews:
    """Tests for embed_reviews()."""

    def test_empty_reviews_raises_value_error(self):
        """embed_reviews([]) must raise ValueError immediately."""
        with pytest.raises(ValueError, match="empty"):
            embed_reviews([], model_name="all-MiniLM-L6-v2")

    def test_output_shape_and_dtype(self):
        """
        embed_reviews returns an ndarray of shape (N, D) where D matches the
        sentence-transformer model's hidden dimension (384 for all-MiniLM-L6-v2).
        """
        reviews = _make_reviews(5)

        # SentenceTransformer is imported *inside* embed_reviews, so we patch
        # it on the sentence_transformers module directly.
        mock_model = MagicMock()
        expected = _fake_embeddings(5, 384)
        mock_model.encode.return_value = expected

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = embed_reviews(reviews, model_name="all-MiniLM-L6-v2")

        assert result.shape == (5, 384), f"Expected (5, 384), got {result.shape}"

    def test_output_is_numpy_array(self):
        """embed_reviews must return a numpy ndarray."""
        reviews = _make_reviews(3)
        mock_model = MagicMock()
        mock_model.encode.return_value = _fake_embeddings(3, 384)

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model

        with patch.dict("sys.modules", {"sentence_transformers": mock_st_module}):
            result = embed_reviews(reviews)

        assert isinstance(result, np.ndarray)

    def test_import_error_propagated(self):
        """ImportError from sentence-transformers surfaces as ImportError."""
        import sys
        # Hide the module entirely
        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore

        try:
            with pytest.raises(ImportError, match="sentence-transformers"):
                embed_reviews(_make_reviews(2))
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original


# ---------------------------------------------------------------------------
# cluster_reviews tests
# ---------------------------------------------------------------------------

class TestClusterReviews:
    """Tests for cluster_reviews()."""

    def _run_cluster(self, embeddings: np.ndarray, reviews: list[dict], **cfg_overrides):
        """Run cluster_reviews with mocked UMAP + HDBSCAN."""
        n = embeddings.shape[0]
        config = _cluster_config(**cfg_overrides)

        # Synthetic UMAP output: passthrough (already low-dimensional)
        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = embeddings[:, :2].astype(np.float32)

        mock_umap = MagicMock()
        mock_umap.UMAP.return_value = mock_umap_instance

        return mock_umap, config

    def test_produces_expected_cluster_count(self):
        """
        With synthetic data in two tight groups, cluster_reviews should return
        at least 1 cluster.
        """
        # Build two tight clusters in 2-D space
        rng = np.random.RandomState(0)
        group_a = rng.randn(15, 384) * 0.01 + np.array([10.0] + [0.0] * 383)
        group_b = rng.randn(15, 384) * 0.01 + np.array([-10.0] + [0.0] * 383)
        embeddings = np.vstack([group_a, group_b]).astype(np.float32)
        reviews = _make_reviews(30)

        # Mock UMAP to return obvious 2-D clusters
        reduced_a = rng.randn(15, 2) * 0.01 + np.array([5.0, 0.0])
        reduced_b = rng.randn(15, 2) * 0.01 + np.array([-5.0, 0.0])
        reduced = np.vstack([reduced_a, reduced_b]).astype(np.float32)

        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = reduced

        mock_umap_mod = MagicMock()
        mock_umap_mod.UMAP.return_value = mock_umap_instance

        # Labels: first 15 → cluster 0, next 15 → cluster 1
        labels = np.array([0] * 15 + [1] * 15)
        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = labels

        mock_hdbscan_mod = MagicMock()
        mock_hdbscan_mod.HDBSCAN.return_value = mock_hdbscan_instance

        config = _cluster_config()

        with patch.dict("sys.modules", {"umap": mock_umap_mod, "hdbscan": mock_hdbscan_mod}):
            result = cluster_reviews(embeddings, reviews, config)

        assert len(result) == 2, f"Expected 2 clusters, got {len(result)}"
        assert all(isinstance(c, RawCluster) for c in result)

    def test_zero_clusters_triggers_runtime_error(self):
        """
        If HDBSCAN assigns all reviews to noise (label = -1), cluster_reviews
        must raise RuntimeError.
        """
        embeddings = _fake_embeddings(10)
        reviews = _make_reviews(10)

        # All labels are -1 (noise)
        noise_labels = np.array([-1] * 10)

        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = embeddings[:, :2]

        mock_umap_mod = MagicMock()
        mock_umap_mod.UMAP.return_value = mock_umap_instance

        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = noise_labels

        mock_hdbscan_mod = MagicMock()
        mock_hdbscan_mod.HDBSCAN.return_value = mock_hdbscan_instance

        config = _cluster_config()

        with patch.dict("sys.modules", {"umap": mock_umap_mod, "hdbscan": mock_hdbscan_mod}):
            with pytest.raises(RuntimeError, match="0 clusters"):
                cluster_reviews(embeddings, reviews, config)

    def test_max_themes_cap_respected(self):
        """cluster_reviews must return at most max_themes clusters."""
        n = 50
        embeddings = _fake_embeddings(n)
        reviews = _make_reviews(n)

        # Simulate 10 equal-size clusters
        labels = np.array([i % 10 for i in range(n)])

        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = embeddings[:, :2]

        mock_umap_mod = MagicMock()
        mock_umap_mod.UMAP.return_value = mock_umap_instance

        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = labels

        mock_hdbscan_mod = MagicMock()
        mock_hdbscan_mod.HDBSCAN.return_value = mock_hdbscan_instance

        max_themes = 3
        config = _cluster_config(max_themes=max_themes)

        with patch.dict("sys.modules", {"umap": mock_umap_mod, "hdbscan": mock_hdbscan_mod}):
            result = cluster_reviews(embeddings, reviews, config)

        assert len(result) <= max_themes, (
            f"Expected at most {max_themes} clusters, got {len(result)}"
        )

    def test_clusters_sorted_by_size_descending(self):
        """Clusters must be sorted largest-first."""
        n = 30
        embeddings = _fake_embeddings(n)
        reviews = _make_reviews(n)

        # 3 clusters of sizes 15, 10, 5
        labels = np.array([0] * 15 + [1] * 10 + [2] * 5)

        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = embeddings[:, :2]
        mock_umap_mod = MagicMock()
        mock_umap_mod.UMAP.return_value = mock_umap_instance

        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = labels
        mock_hdbscan_mod = MagicMock()
        mock_hdbscan_mod.HDBSCAN.return_value = mock_hdbscan_instance

        config = _cluster_config()

        with patch.dict("sys.modules", {"umap": mock_umap_mod, "hdbscan": mock_hdbscan_mod}):
            result = cluster_reviews(embeddings, reviews, config)

        sizes = [c.review_count for c in result]
        assert sizes == sorted(sizes, reverse=True), f"Clusters not sorted: {sizes}"

    def test_avg_rating_computed_correctly(self):
        """avg_rating of a cluster should equal mean of its reviews' ratings."""
        n = 4
        reviews = [
            {"review_id": f"r{i}", "text": f"text {i}", "rating": i + 1}
            for i in range(n)  # ratings: 1, 2, 3, 4
        ]
        embeddings = _fake_embeddings(n)

        # All in one cluster
        labels = np.array([0] * n)

        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = embeddings[:, :2]
        mock_umap_mod = MagicMock()
        mock_umap_mod.UMAP.return_value = mock_umap_instance

        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = labels
        mock_hdbscan_mod = MagicMock()
        mock_hdbscan_mod.HDBSCAN.return_value = mock_hdbscan_instance

        config = _cluster_config()

        with patch.dict("sys.modules", {"umap": mock_umap_mod, "hdbscan": mock_hdbscan_mod}):
            result = cluster_reviews(embeddings, reviews, config)

        assert len(result) == 1
        expected_avg = round((1 + 2 + 3 + 4) / 4, 2)
        assert math.isclose(result[0].avg_rating, expected_avg, rel_tol=1e-5)

    def test_noise_reviews_excluded(self):
        """Reviews labeled -1 (HDBSCAN noise) must not appear in any cluster."""
        n = 10
        embeddings = _fake_embeddings(n)
        reviews = _make_reviews(n)

        # 5 in cluster 0, 5 noise
        labels = np.array([0] * 5 + [-1] * 5)

        mock_umap_instance = MagicMock()
        mock_umap_instance.fit_transform.return_value = embeddings[:, :2]
        mock_umap_mod = MagicMock()
        mock_umap_mod.UMAP.return_value = mock_umap_instance

        mock_hdbscan_instance = MagicMock()
        mock_hdbscan_instance.fit_predict.return_value = labels
        mock_hdbscan_mod = MagicMock()
        mock_hdbscan_mod.HDBSCAN.return_value = mock_hdbscan_instance

        config = _cluster_config()

        with patch.dict("sys.modules", {"umap": mock_umap_mod, "hdbscan": mock_hdbscan_mod}):
            result = cluster_reviews(embeddings, reviews, config)

        assert len(result) == 1
        assert result[0].review_count == 5
        for idx in result[0].review_indices:
            assert idx < 5, f"Noise review index {idx} leaked into cluster."
