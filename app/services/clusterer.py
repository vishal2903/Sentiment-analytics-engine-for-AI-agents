import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import umap
from app.config import settings


def reduce_dimensions(vectors: np.ndarray) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=settings.umap_components,
        metric="cosine",
        random_state=42,
        low_memory=False,
    )
    return reducer.fit_transform(vectors)


def cluster_vectors(reduced: np.ndarray, k: int | None = None) -> np.ndarray:
    k = k or settings.cluster_k
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    return km.fit_predict(reduced)


def compute_ari(predicted: np.ndarray, ground_truth: list[str]) -> float:
    encoded = LabelEncoder().fit_transform(ground_truth)
    return adjusted_rand_score(encoded, predicted)


def run_clustering_pipeline(job_id: str) -> None:
    """
    Runs in a separate OS process via ProcessPoolExecutor.
    CPU-bound work MUST NOT run in the async event loop — UMAP takes 60-90s.
    Loads embeddings from .cache/embeddings.npy (offline store, not REST).
    Writes cluster_ids back via psycopg2 Transaction Pooler batch UPDATE.
    """
    import json
    from pathlib import Path
    import psycopg2
    from psycopg2.extras import execute_values
    from app.database import get_client
    from app.config import settings

    db = get_client()

    try:
        db.table("pipeline_jobs").update({"status": "running"}).eq("id", job_id).execute()

        # Offline store pattern: DB = serving layer, cache = ML layer.
        # REST returns vectors as JSON strings + 1000-row cap. Cache avoids both.
        cache_path = Path(__file__).parent.parent.parent / ".cache" / "embeddings.npy"
        vectors = np.load(cache_path)

        # IDs only via paginated REST — lightweight, no embedding column
        ids = []
        page_size = 1000
        offset = 0
        while True:
            result = (
                db.table("conversations")
                .select("id")
                .eq("source", "bitext")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            batch = result.data
            ids.extend(r["id"] for r in batch)
            if len(batch) < page_size:
                break
            offset += page_size

        reduced = reduce_dimensions(vectors)
        labels = cluster_vectors(reduced)

        pairs = [(ids[i], int(labels[i])) for i in range(len(ids))]
        for batch_start in range(0, len(pairs), 500):
            conn = psycopg2.connect(settings.database_url, sslmode="require")
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """UPDATE conversations SET cluster_id = data.cluster_id
                       FROM (VALUES %s) AS data(id, cluster_id)
                       WHERE conversations.id = data.id::uuid""",
                    pairs[batch_start : batch_start + 500],
                    template="(%s, %s)",
                )
            conn.commit()
            conn.close()

        db.table("pipeline_jobs").update(
            {"status": "complete", "completed_at": "now()"}
        ).eq("id", job_id).execute()

    except Exception as exc:
        db.table("pipeline_jobs").update(
            {"status": "failed", "error": str(exc)}
        ).eq("id", job_id).execute()
        raise
