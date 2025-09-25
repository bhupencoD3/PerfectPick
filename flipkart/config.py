import os
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

# Load .env variables
load_dotenv()

logger = get_logger(__name__)


class Config:
    """
    Loads and validates environment variables required for the pipeline.
    """

    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    LLM_MODEL = "llama-3.1-8b-instant"

    # Load sensitive configs from .env
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    @classmethod
    def validate(cls):
        """
        Raises CustomException if any required variable is missing.
        """
        required_vars = {
            "ASTRA_DB_API_ENDPOINT": cls.ASTRA_DB_API_ENDPOINT,
            "ASTRA_DB_APPLICATION_TOKEN": cls.ASTRA_DB_APPLICATION_TOKEN,
            "ASTRA_DB_KEYSPACE": cls.ASTRA_DB_KEYSPACE,
            "GROQ_API_KEY": cls.GROQ_API_KEY,
        }

        missing = [key for key, value in required_vars.items() if not value]

        if missing:
            logger.error(f"Missing environment variables: {missing}")
            raise CustomException(
                "Configuration validation failed",
                f"Missing environment variables: {', '.join(missing)}",
            )

        logger.info("All required environment variables validated successfully.")

        return True
