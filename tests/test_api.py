import concurrent.futures
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_insights_empty():
    with patch("app.routers.insights.get_client") as mock:
        db = MagicMock()
        db.table.return_value.select.return_value.order.return_value.execute.return_value.data = []
        mock.return_value = db
        response = client.get("/insights")
    assert response.status_code == 200
    assert response.json() == []


def test_list_insights_with_data():
    mock_data = [
        {"id": 1, "topic_label": "Refund confusion", "percentage": 22.1, "severity": "HIGH", "week_over_week": 18.3}
    ]
    with patch("app.routers.insights.get_client") as mock:
        db = MagicMock()
        db.table.return_value.select.return_value.order.return_value.execute.return_value.data = mock_data
        mock.return_value = db
        response = client.get("/insights")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["topic_label"] == "Refund confusion"


def test_get_insight_not_found():
    with patch("app.routers.insights.get_client") as mock:
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
        mock.return_value = db
        response = client.get("/insights/999")
    assert response.status_code == 404


def test_ingest_missing_user_message():
    payload = {
        "session_id": "test-001",
        "messages": [{"role": "assistant", "content": "Hello"}]
    }
    with patch("app.routers.ingest.get_client"), patch("app.routers.ingest.get_embedder"):
        response = client.post("/ingest", json=payload)
    assert response.status_code == 422


def test_analyze_returns_202():
    future = concurrent.futures.Future()
    future.set_result(None)
    with patch("app.routers.analyze.get_client") as mock_db, \
         patch("app.routers.analyze._executor") as mock_exec:
        db = MagicMock()
        db.table.return_value.insert.return_value.execute.return_value = MagicMock()
        mock_db.return_value = db
        mock_exec.submit.return_value = future
        response = client.post("/analyze")
    assert response.status_code == 202
    assert "job_id" in response.json()
    assert response.json()["status"] == "pending"
