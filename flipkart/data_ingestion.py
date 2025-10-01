from typing import Optional, List, Iterable
import time
import math
import pandas as pd
from tqdm import tqdm

from utils.logger import get_logger
from utils.custom_exception import CustomException
from utils.validators import DataValidator
from flipkart.config import Config
from flipkart.data_converter import DataConverter

# LangChain integrations
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_astradb import AstraDBVectorStore

logger = get_logger(__name__)


class DataIngestion:

    def __init__(
        self,
        collection_name: str = "flipkart_recommendation",
        batch_size: int = 500,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Validate config
        Config.validate()

        # Initialize embedding + vector store
        try:
            Config.validate()
        except Exception as e:
            raise CustomException("Config validation failed at DataIngestion init", e)

        try:
            # Initialize embedding client
            self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

            self.vector_store = AstraDBVectorStore(
                embedding=self.embedding,
                collection_name=self.collection_name,
                api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
                token=Config.ASTRA_DB_APPLICATION_TOKEN,
                namespace=Config.ASTRA_DB_KEYSPACE,
            )

            logger.info(f"✅ Vector store initialized for collection: {self.collection_name}")

        except Exception as e:
            raise CustomException("Failed to initialize embedding/vector store", e)

    def _chunked(self, iterable: Iterable, size: int):
        """Yield chunks from iterable."""
        it = iter(iterable)
        while True:
            chunk = []
            try:
                for _ in range(size):
                    chunk.append(next(it))
            except StopIteration:
                if chunk:
                    yield chunk
                break
            yield chunk

    def _safe_call_with_retries(self, fn, *args, **kwargs):
        """Simple retry/backoff wrapper."""
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(f"⚠️ Attempt {attempt}/{self.max_retries} failed for {fn.__name__}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        raise CustomException(f"Operation {fn.__name__} failed after retries", last_exc)

    def ingest(self, file_path: str, validate: bool = True, overwrite: bool = False, show_progress: bool = True):
        """Main ingestion pipeline."""
        try:
            logger.info(f"📂 Loading data from {file_path}")
            df = pd.read_csv(file_path)

            if validate:
                logger.info("🧹 Running dataframe validation")
                df = DataValidator(df).run_all_validations(drop_missing=True)

            logger.info("🔁 Converting dataframe to LangChain Documents")
            docs = DataConverter(df).convert()
            total_docs = len(docs)
            logger.info(f"Converted {total_docs} rows into Documents")

            # Overwrite old data if requested
            if overwrite:
                logger.info("🧨 Overwrite requested: attempting to clear collection")
                try:
                    self.vector_store.delete_collection()
                    logger.info(f"Collection '{self.collection_name}' cleared successfully")
                except Exception as e:
                    logger.warning(f"Failed to clear collection: {e}")

            # Upload in batches
            logger.info(f"🚀 Ingesting {total_docs} documents in batches of {self.batch_size}")
            iterator = (
                tqdm(self._chunked(docs, self.batch_size), total=math.ceil(total_docs / self.batch_size))
                if show_progress else self._chunked(docs, self.batch_size)
            )

            for chunk in iterator:
                self._safe_call_with_retries(self.vector_store.add_documents, documents=chunk)

            logger.info(f"✅ Ingestion complete. {total_docs} docs uploaded to '{self.collection_name}'")
            return self.vector_store

        except Exception as e:
            logger.exception("❌ Data ingestion pipeline failed")
            raise CustomException("Data ingestion failed", e)
