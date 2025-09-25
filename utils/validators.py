import pandas as pd
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class DataValidator:
    REQUIRED_COLUMNS = ['product_title', 'review']

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise CustomException("Input must be a pandas DataFrame")
        self.df = df.copy()
    
    def validate_columns(self):
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_cols:
            raise CustomException(f"Missing required columns: {missing_cols}")
        logger.info("All required columns are present")
        return True

    def validate_missing_values(self, drop_missing=True):
        invalid_rows = self.df[self.REQUIRED_COLUMNS].isnull().any(axis=1) | \
                       (self.df[self.REQUIRED_COLUMNS].applymap(lambda x: str(x).strip() == '')).any(axis=1)
        num_invalid = invalid_rows.sum()
        if num_invalid > 0:
            logger.warning(f"Found {num_invalid} rows with missing/empty values")
            if drop_missing:
                self.df = self.df[~invalid_rows].reset_index(drop=True)
                logger.info(f"Dropped {num_invalid} invalid rows")
        else:
            logger.info("No missing or empty values found")
        return self.df

    def validate_data_types(self):
        for col in self.REQUIRED_COLUMNS:
            self.df[col] = self.df[col].astype(str)
        logger.info("Data types validated and converted to string")
        return self.df

    def run_all_validations(self, drop_missing=True):
        try:
            self.validate_columns()
            self.validate_missing_values(drop_missing)
            self.validate_data_types()
            logger.info("All validations completed successfully")
            return self.df
        except Exception as e:
            raise CustomException("Data validation failed", e)
