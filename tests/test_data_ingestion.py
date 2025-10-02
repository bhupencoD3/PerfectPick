import pytest
from unittest.mock import patch, MagicMock

@patch("flipkart.data_ingestion.Config")  # patch the Config inside data_ingestion
@patch("flipkart.data_converter.DataConverter.convert")
@patch("flipkart.data_ingestion.AstraDBVectorStore")
@patch("flipkart.data_ingestion.HuggingFaceEndpointEmbeddings")
def test_data_ingestion(mock_hf, mock_store, mock_convert, mock_config, tmp_path):
    from perfectpick.data_ingestion import DataIngestion

    # Set dummy values for Config attributes
    mock_config.ASTRA_DB_API_ENDPOINT = "https://dummy.endpoint"
    mock_config.ASTRA_DB_APPLICATION_TOKEN = "dummy-token"
    mock_config.ASTRA_DB_KEYSPACE = "dummy-keyspace"
    mock_config.GROQ_API_KEY = "dummy-groq-key"
    mock_config.validate.return_value = None  # skip actual validation

    # dummy CSV
    csv_file = tmp_path / "dummy_flipkart_reviews.csv"
    csv_file.write_text("product_title,review\nProduct A,Good product\nProduct B,Not bad\nProduct C,Average")

    mock_convert.return_value = []
    mock_store_instance = MagicMock()
    mock_store.return_value = mock_store_instance

    ingestion = DataIngestion()
    vector_store = ingestion.ingest(file_path=str(csv_file), overwrite=False)

    assert vector_store is mock_store_instance
