import pandas as pd
import pytest
from utils.validators import DataValidator
from utils.custom_exception import CustomException

@pytest.fixture
def dummy_df():
    return pd.DataFrame({
        "product_title": ["Product A", "Product B", "Product C"],
        "review": ["Good product", "Not bad", "Average"]
    })

def test_validate_columns(dummy_df):
    validator = DataValidator(dummy_df)
    assert validator.validate_columns() is True

def test_validate_missing_values(dummy_df):
    validator = DataValidator(dummy_df)
    cleaned_df = validator.validate_missing_values()
    assert len(cleaned_df) == 3

def test_validate_data_types(dummy_df):
    validator = DataValidator(dummy_df)
    df2 = validator.validate_data_types()
    assert df2['product_title'].dtype == object
    assert df2['review'].dtype == object

def test_run_all_validations(dummy_df):
    validator = DataValidator(dummy_df)
    df2 = validator.run_all_validations()
    assert len(df2) == 3

def test_invalid_input_type():
    with pytest.raises(CustomException):
        DataValidator("not a dataframe")
