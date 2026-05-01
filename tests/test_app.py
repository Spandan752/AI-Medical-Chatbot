import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    with patch("app.embedding_model"), \
         patch("app.docsearch"), \
         patch("app.rag_chain"):
        from app import app
        return TestClient(app)


@pytest.mark.unit
def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data
    assert "environment" in data


@pytest.mark.unit
def test_chat_endpoint_success(client):
    """Test chat endpoint with valid request."""
    with patch("app.rag_chain") as mock_chain:
        mock_chain.invoke.return_value = "Test response from chatbot"
        
        response = client.post(
            "/chat",
            json={"input": "What is diabetes?"}
        )
        
        assert response.status_code == 200
        assert response.json() == {"response": "Test response from chatbot"}


@pytest.mark.unit
def test_chat_endpoint_empty_input(client):
    """Test chat endpoint with empty input."""
    with patch("app.rag_chain") as mock_chain:
        response = client.post(
            "/chat",
            json={"input": "   "}
        )
        
        assert response.status_code == 400


@pytest.mark.unit
def test_chat_endpoint_invalid_input(client):
    """Test chat endpoint with invalid input."""
    response = client.post(
        "/chat",
        json={"invalid_field": "test"}
    )
    
    assert response.status_code == 422  # Validation error


@pytest.mark.unit
def test_chat_endpoint_error_handling(client):
    """Test chat endpoint error handling."""
    with patch("app.rag_chain") as mock_chain:
        mock_chain.invoke.side_effect = Exception("API Error")
        
        response = client.post(
            "/chat",
            json={"input": "What is diabetes?"}
        )
        
        assert response.status_code == 500
