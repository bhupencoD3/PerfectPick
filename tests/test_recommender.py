# tests/test_recommender.py

import pytest
from unittest.mock import patch, MagicMock
from flipkart.recommender import Recommender
from utils.custom_exception import CustomException

@patch("flipkart.recommender.DataIngestion")
def test_recommender_initialization(mock_data_ingestion):
    # Mock the vector store attribute
    mock_vector_store = MagicMock()
    mock_data_ingestion.return_value.vector_store = mock_vector_store

    # Instantiate recommender
    recommender = Recommender(collection_name="test-collection")

    # Check that vector store is set correctly
    assert recommender.vector_store == mock_vector_store
    mock_data_ingestion.assert_called_once_with(collection_name="test-collection")


@patch("flipkart.recommender.DataIngestion")
def test_recommend_success(mock_data_ingestion):
    # Mock the vector store similarity search
    mock_vector_store = MagicMock()
    mock_result = MagicMock()
    mock_result.id = "123"
    mock_result.page_content = "Test product"
    mock_result.score = 0.95
    mock_vector_store.similarity_search.return_value = [mock_result]

    mock_data_ingestion.return_value.vector_store = mock_vector_store
    recommender = Recommender(collection_name="test-collection")

    # Call recommend
    query = "Test query"
    recs = recommender.recommend(query, top_k=1)

    # Validate output
    assert len(recs) == 1
    assert recs[0]["id"] == "123"
    assert recs[0]["content"] == "Test product"
    assert recs[0]["score"] == 0.95
    mock_vector_store.similarity_search.assert_called_once_with(query, k=1)


@patch("flipkart.recommender.DataIngestion")
def test_recommend_failure_raises_custom_exception(mock_data_ingestion):
    # Mock vector store to raise exception
    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search.side_effect = Exception("Some error")
    mock_data_ingestion.return_value.vector_store = mock_vector_store
    recommender = Recommender(collection_name="test-collection")

    with pytest.raises(CustomException) as exc_info:
        recommender.recommend("query")
    
    assert "Recommendation query failed" in str(exc_info.value)
