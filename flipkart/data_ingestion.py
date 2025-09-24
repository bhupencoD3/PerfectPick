from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from flipkart.data_converter import DataConverter
from flipkart.config import Config
from utils.custom_exception import CustomException  

class DataIngestion:
    def __init__(self):
        try:
            self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)
            self.vector_store = AstraDBVectorStore(
                embedding=self.embedding,
                collection_name="flipkart-recommendation",
                api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
                token=Config.ASTRA_DB_APPLICATION_TOKEN,
                namespace=Config.ASTRA_DB_KEYSPACE,
            )
        except Exception as e:
            raise CustomException("Failed to initialize DataIngestion", e)

    def ingest(self, file_path: str = "data/flipkart_product_review.csv", load_existing: bool = True):
        if load_existing:
            # Optionally: check if collection already has docs
            return self.vector_store

        try:
            docs = DataConverter(file_path).convert()
            self.vector_store.add_documents(documents=docs)
            return self.vector_store
        except Exception as e:
            raise CustomException("Data ingestion failed", e)
