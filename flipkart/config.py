import os
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class Config:
    """
    Singleton-style configuration loader and validator.
    Ensures all critical environment variables are loaded once at import time.
    """

    # === Constants ===
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "llama-3.1-8b-instant"
    DATA_FILE_PATH = "/app/data/Flipkart_Mobiles.csv"

    # === Environment Variables ===
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    DB_URL = os.getenv("DB_URL")

    _validated = False  # internal flag to prevent double validation

    @classmethod
    def validate(cls):
        """Ensure all required variables are available (only once)."""
        if cls._validated:
            return True  # skip if already validated

        required_vars = {
            "ASTRA_DB_API_ENDPOINT": cls.ASTRA_DB_API_ENDPOINT,
            "ASTRA_DB_APPLICATION_TOKEN": cls.ASTRA_DB_APPLICATION_TOKEN,
            "ASTRA_DB_KEYSPACE": cls.ASTRA_DB_KEYSPACE,
            "GROQ_API_KEY": cls.GROQ_API_KEY,
        }

        missing = [key for key, value in required_vars.items() if not value]
        if missing:
            raise CustomException(
                "Config validation failed", f"Missing environment variables: {', '.join(missing)}"
            )

        cls._validated = True
        logger.info("✅ Configuration successfully loaded and validated.")
        return True
