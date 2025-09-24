import pandas as pd
from langchain_core.documents import Document
from typing import List

class DataConverter:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def convert(self) -> List[Document]:
        df = pd.read_csv(self.file_path)

        # Check required columns
        required_cols = ['product_title', 'review']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        docs = [
            Document(page_content=row.review, metadata={'product_name': row.product_title})
            for row in df.itertuples(index=False)
        ]

        return docs
