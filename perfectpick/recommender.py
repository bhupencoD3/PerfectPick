# flipkart/recommender.py

from typing import List, Optional
from utils.custom_exception import CustomException
from perfectpick.data_ingestion import DataIngestion
from utils.logger import get_logger

logger = get_logger(__name__, log_file="recommender.log")


class Recommender:
    def __init__(self, collection_name: str = "flipkart_recommendation"):
        self.collection_name = collection_name
        try:
            # Initialize DataIngestion to access vector store
            self.data_ingestion = DataIngestion(collection_name=self.collection_name)
            self.vector_store = self.data_ingestion.vector_store
        except Exception as e:
            raise CustomException("Failed to initialize Recommender", e)

    def recommend(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[dict]:
        """
        Given a product query string, returns top-K similar products from vector store.

        Returns:
            List of dicts containing at least 'id', 'content', and 'score'.
        """
        try:
            # Call vector store similarity search
            results = self.vector_store.similarity_search(query, k=top_k)
            
            # Ensure output is standardized
            recommendations = []
            for r in results:
                recommendations.append({
                    "id": getattr(r, "id", None),
                    "content": getattr(r, "page_content", str(r)),
                    "score": getattr(r, "score", None)
                })
            return recommendations

        except Exception as e:
            logger.exception("Error fetching recommendations")
            raise CustomException("Recommendation query failed", e)


if __name__ == "__main__":
    # Quick sanity check
    recommender = Recommender()
    sample_query = "Wireless Bluetooth Headphones"
    recs = recommender.recommend(sample_query, top_k=3)
    for i, r in enumerate(recs, 1):
        print(f"{i}. {r['content']} (score: {r['score']})")
