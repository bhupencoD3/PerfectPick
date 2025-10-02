import chardet
import pandas as pd
from utils.logger import get_logger
from utils.custom_exception import CustomException
from perfectpick.data_ingestion import DataIngestion

logger = get_logger(__name__)

class PerfectPickService:
    def __init__(self, file_path="data/Flipkart_Mobiles_cleaned.csv", overwrite=False):
        self.file_path = file_path
        self.overwrite = overwrite
        self.ingestion = DataIngestion(collection_name="flipkart_recommendation")
        self.vector_store = self.ingestion.vector_store

    def load_data(self):
        """Detect encoding + load CSV safely."""
        try:
            with open(self.file_path, "rb") as f:
                result = chardet.detect(f.read())
            encoding = result["encoding"]
            df = pd.read_csv(self.file_path, encoding=encoding, on_bad_lines="skip", low_memory=False)
            logger.info(f"Loaded CSV with {len(df)} rows")
            return df
        except Exception as e:
            raise CustomException("Failed to load data", e)

    def run_service(self):
        """Run ingestion service."""
        df = self.load_data()
        self.vector_store = self.ingestion.ingest(self.file_path, overwrite=self.overwrite, show_progress=True)
        logger.info("✅ Flipkart recommendation ingestion completed successfully")
        return self.vector_store
