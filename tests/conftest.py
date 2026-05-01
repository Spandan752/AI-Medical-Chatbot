import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_query():
    """Fixture for sample medical query."""
    return "What are the symptoms of diabetes?"

@pytest.fixture
def sample_documents():
    """Fixture for sample documents."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="Diabetes is a chronic condition affecting blood sugar levels.",
            metadata={"source": "medical_db_1.pdf"}
        ),
        Document(
            page_content="Common symptoms include increased thirst and fatigue.",
            metadata={"source": "medical_db_2.pdf"}
        ),
    ]

@pytest.fixture
def mock_settings():
    """Fixture for mocked settings."""
    with patch('config.settings') as mock:
        mock.PINECONE_INDEX_NAME = "medical-chatbot"
        mock.TOP_K_RETRIEVAL = 3
        mock.ENVIRONMENT = "development"
        mock.DEBUG = True
        mock.HOST = "0.0.0.0"
        mock.PORT = 8000
        yield mock
