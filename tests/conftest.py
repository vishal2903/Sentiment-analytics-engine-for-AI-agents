import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_db():
    with patch("app.database.get_client") as mock:
        db = MagicMock()
        mock.return_value = db
        yield db
