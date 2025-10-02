"""
Entry point to ingest Flipkart mobiles data into the vector store.
"""

import argparse
from perfectpick.service import PerfectPickService
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Flipkart Recommendation Data Ingestion Pipeline")
    parser.add_argument("--file", default="data/Flipkart_Mobiles_cleaned.csv", help="Path to CSV file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing collection")
    args = parser.parse_args()

    try:
        logger.info("🚀 Starting Flipkart Recommendation Data Pipeline")
        service = PerfectPickService(
            file_path=args.file,
            overwrite=args.overwrite
        )
        vector_store = service.run_service()
        total_docs = getattr(vector_store, "count", "unknown")  # optional: show number of docs ingested
        logger.info(f"🏁 Pipeline completed successfully. Total docs ingested: {total_docs}")

    except CustomException as ce:
        logger.error(f"Pipeline failed: {ce}")
    except Exception as e:
        logger.exception("Unexpected error occurred during pipeline execution")

if __name__ == "__main__":
    main()
