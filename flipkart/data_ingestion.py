from typing import Optional, List, Iterable
import time
import math

import pandas as pd
from tqdm import tqdm

from utils.logger import get_logger
from utils.custom_exception import CustomException
from flipkart.config import Config
from flipkart.data_converter import DataConverter
from utils.validators import DataValidator

# Import vector/embedding backends
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_astradb import AstraDBVectorStore

logger = get_logger(__name__, log_file="data_ingestion.log")


class DataIngestion:

    def __init__(
        self,
        collection_name: str = "flipkart-recommendation",
        batch_size: int = 500,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Validate config early (fail-fast)
        try:
            Config.validate()
        except Exception as e:
            raise CustomException("Config validation failed at DataIngestion init", e)

        try:
            # Initialize embedding client
            self.embedding = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

            # Initialize AstraDB vector store
            self.vector_store = AstraDBVectorStore(
                embedding=self.embedding,
                collection_name=self.collection_name,
                api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
                token=Config.ASTRA_DB_APPLICATION_TOKEN,
                namespace=Config.ASTRA_DB_KEYSPACE,
            )
        except Exception as e:
            raise CustomException("Failed to initialize embedding/vector store", e)

    def _chunked(self, iterable: Iterable, size: int) -> Iterable[List]:
        """Yield successive chunks from iterable of given size."""
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
        """Run function with simple retry/backoff logic for transient errors."""
        last_exc = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"Attempt {attempt}/{self.max_retries} failed for {fn.__name__}: {e}"
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff * attempt)
        # if we get here, all retries failed
        raise CustomException(f"Operation {fn.__name__} failed after retries", last_exc)

    def _attempt_clear_collection(self):
        # Try common names for clearing/deleting collection
        for method_name in ("delete_collection", "drop_collection", "clear_collection", "clear"):
            method = getattr(self.vector_store, method_name, None)
            if callable(method):
                logger.info(f"Attempting to clear collection using {method_name}()")
                return self._safe_call_with_retries(method)
        # If none found, log notice and continue (no destructive action)
        logger.info(
            "No direct collection delete/clear method found on vector_store; skipping overwrite deletion."
        )
        return None

    def _get_collection_size(self) -> Optional[int]:
        for method_name in ("count_documents", "get_count", "collection_size", "size", "count"):
            method = getattr(self.vector_store, method_name, None)
            if callable(method):
                try:
                    size = method()
                    logger.info(f"Collection size obtained via {method_name}(): {size}")
                    return int(size)
                except Exception:
                    logger.debug(f"Method {method_name} exists but failed to return size.")
        logger.debug("No collection size method available on vector_store.")
        return None

    def ingest(
        self,
        file_path: str,
        validate: bool = True,
        overwrite: bool = False,
        show_progress: bool = True,
    ):
        try:
            # Read CSV
            logger.info(f"Loading data from: {file_path}")
            df = pd.read_csv(file_path)

            # Run validation if requested
            if validate:
                logger.info("Validating dataframe with DataValidator")
                validator = DataValidator(df)
                df = validator.run_all_validations(drop_missing=True)

            # Convert to LangChain Documents
            logger.info("Converting dataframe to Documents via DataConverter")
            docs = DataConverter(file_path).convert()

            # Overwrite behavior: attempt to clear existing collection if requested
            if overwrite:
                logger.info("Overwrite requested: attempting to clear existing collection")
                self._attempt_clear_collection()

            # Basic idempotency: if collection already has items and overwrite is False, skip ingestion
            existing_count = self._get_collection_size()
            if existing_count and existing_count > 0 and not overwrite:
                logger.info(
                    f"Collection '{self.collection_name}' already contains {existing_count} items. "
                    "Skipping ingestion because overwrite=False."
                )
                return self.vector_store

            # Batch add documents
            total_docs = len(docs)
            logger.info(f"Starting ingestion of {total_docs} documents in batches of {self.batch_size}")

            # tqdm optional
            iterator = (
                tqdm(self._chunked(docs, self.batch_size), total=math.ceil(total_docs / self.batch_size))
                if show_progress
                else self._chunked(docs, self.batch_size)
            )

            for chunk in iterator:
                # Use safe call wrapper to add documents in a retriable manner
                logger.debug(f"Uploading batch of size {len(chunk)}")
                self._safe_call_with_retries(self.vector_store.add_documents, documents=chunk)

            logger.info(f"Ingestion complete. {total_docs} documents uploaded to collection '{self.collection_name}'")
            return self.vector_store

        except CustomException:
            # Re-raise known custom exceptions
            raise
        except Exception as e:
            logger.exception("Unhandled exception during data ingestion")
            raise CustomException("Data ingestion pipeline failed", e)
