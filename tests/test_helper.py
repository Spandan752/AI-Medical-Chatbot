import pytest
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock
from src.helper import filter_page_content, text_splitter


@pytest.mark.unit
def test_filter_page_content(sample_documents):
    """Test filter_page_content function."""
    result = filter_page_content(sample_documents)
    
    assert len(result) == 2
    assert all(isinstance(doc, Document) for doc in result)
    assert result[0].metadata == {"source": "medical_db_1.pdf"}
    assert result[1].metadata == {"source": "medical_db_2.pdf"}


@pytest.mark.unit
def test_filter_page_content_empty_list():
    """Test filter_page_content with empty list."""
    result = filter_page_content([])
    assert result == []


@pytest.mark.unit
def test_text_splitter(sample_documents):
    """Test text_splitter function."""
    result = text_splitter(sample_documents)
    
    assert len(result) > 0
    assert all(isinstance(chunk, Document) for chunk in result)


@pytest.mark.unit
def test_text_splitter_preserves_metadata(sample_documents):
    """Test that text_splitter preserves document metadata."""
    result = text_splitter(sample_documents)
    
    # Should have metadata from original documents
    assert all("source" in chunk.metadata for chunk in result)


@pytest.mark.unit
@patch("src.helper.HuggingFaceEmbeddings")
def test_download_embeddings(mock_embeddings):
    """Test download_embeddings function."""
    from src.helper import download_embeddings
    
    mock_embeddings.return_value = MagicMock()
    
    result = download_embeddings()
    
    mock_embeddings.assert_called_once()