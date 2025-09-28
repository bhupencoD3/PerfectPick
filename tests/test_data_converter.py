import pytest
import pandas as pd
from flipkart.data_converter import DataConverter

@pytest.fixture
def dummy_csv(tmp_path):
    csv_file = tmp_path / "dummy_flipkart_reviews.csv"
    df = pd.DataFrame({
        "product_title": ["Product A", "Product B", "Product C"],
        "review": ["Good product", "Not bad", "Average"]
    })
    df.to_csv(csv_file, index=False)
    return str(csv_file)

def test_data_converter(dummy_csv):
    converter = DataConverter(dummy_csv)
    docs = converter.convert()
    assert len(docs) == 3
