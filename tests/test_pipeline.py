import numpy as np
import pytest


def test_embed_batch_returns_correct_shape():
    from unittest.mock import patch, MagicMock
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1536) for _ in range(3)]

    with patch("app.services.embedder.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = mock_response
        from app.services.embedder import OpenAIEmbedder
        embedder = OpenAIEmbedder()
        result = embedder.embed_batch(["a", "b", "c"])

    assert len(result) == 3
    assert len(result[0]) == 1536


def test_cluster_vectors_returns_correct_count():
    from app.services.clusterer import cluster_vectors
    rng = np.random.default_rng(42)
    reduced = rng.random((100, 5)).astype(np.float32)
    labels = cluster_vectors(reduced, k=5)
    assert len(labels) == 100
    assert len(set(labels)) == 5


def test_ari_perfect_score():
    from app.services.clusterer import compute_ari
    labels = np.array([0, 0, 1, 1, 2, 2])
    ground_truth = ["a", "a", "b", "b", "c", "c"]
    ari = compute_ari(labels, ground_truth)
    assert ari == pytest.approx(1.0)


def test_ari_random_score():
    from app.services.clusterer import compute_ari
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 27, size=1000)
    ground_truth = [str(x) for x in rng.integers(0, 27, size=1000)]
    ari = compute_ari(labels, ground_truth)
    assert ari < 0.1


def test_sample_cluster_conversations_returns_n():
    from app.services.labeler import sample_cluster_conversations
    rng = np.random.default_rng(42)
    rows = [
        {"cluster_id": 0, "user_message": f"msg {i}", "embedding": rng.random(1536).tolist()}
        for i in range(20)
    ]
    samples = sample_cluster_conversations(0, rows, n=5)
    assert len(samples) == 5
