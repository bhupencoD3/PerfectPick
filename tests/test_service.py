import pytest
from unittest.mock import patch, MagicMock
from flipkart.service import FlipkartRecommendationService

@patch("flipkart.service.DataIngestion")
def test_service_init(mock_ingestion):
    mock_instance = MagicMock()
    mock_instance.vector_store = "dummy_vector_store"
    mock_ingestion.return_value = mock_instance

    service = FlipkartRecommendationService()
    assert service.vector_store == "dummy_vector_store"
