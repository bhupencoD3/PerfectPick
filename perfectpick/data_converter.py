from langchain_core.documents import Document
from typing import List, Union
import pandas as pd
from utils.logger import get_logger
from utils.custom_exception import CustomException
import chardet
import re

logger = get_logger(__name__)

class DataConverter:
    def __init__(self, source: Union[str, pd.DataFrame]):
        self.df = None
        self.file_path = None
        if isinstance(source, pd.DataFrame):
            self.df = source.copy()
        elif isinstance(source, str):
            self.file_path = source
        else:
            raise ValueError("source must be a pandas DataFrame or file path string")

    def _clean_price(self, value):
        """Clean and normalize price strings safely."""
        if pd.isna(value):
            return None
        try:
            # Remove all non-digit or dot characters (like commas, ₹, spaces)
            cleaned = re.sub(r"[^\d.]", "", str(value))
            if cleaned.strip() == "":
                return None
            return float(cleaned)
        except Exception:
            return None

    def convert(self) -> List[Document]:
        try:
            # 🧩 Step 1: Load CSV if not dataframe
            if self.df is None:
                with open(self.file_path, 'rb') as f:
                    result = chardet.detect(f.read())
                encoding = result['encoding']
                self.df = pd.read_csv(self.file_path, encoding=encoding, on_bad_lines='skip', low_memory=False)

            # 🧠 Step 2: Validate required columns
            required_cols = ['Model', 'Selling Price']
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # 🧹 Step 3: Drop missing critical values
            self.df.dropna(subset=required_cols, inplace=True)

            # 🧽 Step 4: Clean up the price columns
            for col in ["Selling Price", "Original Price"]:
                if col in self.df.columns:
                    self.df[col] = (
                        self.df[col]
                        .astype(str)
                        .str.replace(r"[^\d.]", "", regex=True)  # Remove ₹, commas, text
                        .replace("", "0")
                    )

            # 🧩 Step 5: Available spec columns
            spec_cols = ['Brand', 'Color', 'Memory', 'Storage', 'Rating', 'Original Price']
            available_spec_cols = [col for col in spec_cols if col in self.df.columns]

            docs = []
            skipped_rows = 0

            # ⚙️ Step 6: Convert rows → LangChain Documents
            for i, row in self.df.iterrows():
                model = row.get("Model", "Unknown")
                selling_price = self._clean_price(row.get("Selling Price"))
                original_price = self._clean_price(row.get("Original Price"))

                # 💥 Skip invalid rows
                if not selling_price or selling_price <= 0:
                    logger.warning(f"Skipping row {i}: Invalid Selling Price for model '{model}' → {row.get('Selling Price')}")
                    skipped_rows += 1
                    continue

                # Format prices
                selling_price_formatted = f"₹{selling_price:,.0f}"
                original_price_str = f"{original_price:,.0f}" if original_price and original_price > 0 else "N/A"

                # 📋 Build specs string
                specs = []
                for col in available_spec_cols:
                    val = row.get(col, "N/A")
                    if col == "Original Price":
                        val = original_price_str
                    specs.append(f"{col}: {val}")

                page_content = f"Brand: {row.get('Brand', 'N/A')}, Model: {model}, Selling Price: {selling_price_formatted}, {', '.join(specs)}"

                metadata = {
                    'model': str(model),
                    'price': selling_price,
                    'brand': str(row.get('Brand', 'Unknown')),
                    'rating': str(row.get('Rating', 'N/A'))
                }

                docs.append(Document(page_content=page_content, metadata=metadata))

            # ✅ Final Log
            logger.info(f"Converted {len(docs)} rows into Documents, skipped {skipped_rows} invalid rows")
            return docs

        except Exception as e:
            logger.error("Failed to convert CSV to Documents")
            raise CustomException("Data conversion failed", e)
