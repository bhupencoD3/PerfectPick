# flipkart/service.py

from flipkart.config import Config
from flipkart.data_converter import DataConverter
from utils.validators import DataValidator
from flipkart.data_ingestion import DataIngestion
from utils.logger import get_logger
from utils.custom_exception import CustomException
import pandas as pd

logger = get_logger(__name__)

class FlipkartRecommendationService:
    def __init__(self, file_path: str = "data/flipkart_product_review.csv", load_existing: bool = True):
        self.file_path = file_path
        self.load_existing = load_existing
        self.ingestion = DataIngestion()
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            return self.df
        except Exception as e:
            logger.error("Failed to load CSV")
            raise CustomException("Loading CSV failed", e)

    def validate_data(self):
        try:
            validator = DataValidator(self.df)
            self.df = validator.run_all_validations()
            return self.df
        except Exception as e:
            logger.error("Data validation failed")
            raise CustomException("Data validation failed", e)

    def convert_and_ingest(self):
        try:
            docs = DataConverter(self.file_path).convert()
            vector_store = self.ingestion.ingest(load_existing=self.load_existing)
            if not self.load_existing:
                vector_store.add_documents(docs)
            return vector_store
        except Exception as e:
            logger.error("Data conversion or ingestion failed")
            raise CustomException("Data ingestion failed", e)

    def run_service(self):
        self.load_data()
        self.validate_data()
        vector_store = self.convert_and_ingest()
        logger.info("Flipkart recommendation pipeline completed successfully")
        return vector_store
