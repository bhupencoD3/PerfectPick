import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into environment

class Config:
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Default values set explicitly
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    @classmethod
    def validate(cls):
        required = [
            "ASTRA_DB_API_ENDPOINT",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE",
            "GROQ_API_KEY",
        ]
        missing = [r for r in required if getattr(cls, r) is None]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
