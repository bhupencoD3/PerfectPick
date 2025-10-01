import pandas as pd
from io import StringIO
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class DataValidator:
    REQUIRED_COLUMNS = ['Model', 'Selling Price']
    OPTIONAL_COLUMNS = ['Brand', 'Color', 'Memory', 'Storage', 'Rating', 'Original Price']

    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataValidator with a DataFrame.
        Args:
            df: pandas DataFrame from Flipkart_Mobiles.csv.
        Raises:
            CustomException: If input is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error({"message": "Invalid input type", "type": str(type(df))})
            raise CustomException("Invalid input", "Input must be a pandas DataFrame")
        self.df = df.copy()
        logger.debug({"message": "DataValidator initialized with DataFrame"})

    def run_all_validations(self, drop_missing=True, chunksize=10000) -> pd.DataFrame:
        """
        Run all validations on the DataFrame, processing in chunks for scalability.
        Args:
            drop_missing: If True, drop rows with missing/empty required columns.
            chunksize: Number of rows to process per chunk for large datasets.
        Returns:
            pd.DataFrame: Validated DataFrame.
        Raises:
            CustomException: If validation fails.
        """
        try:
            self._validate_columns()
            if chunksize:
                chunks = []
                csv_buffer = self.df if isinstance(self.df, str) else StringIO(self.df.to_csv(index=False))
                for chunk in pd.read_csv(csv_buffer, 
                                         chunksize=chunksize, 
                                         encoding="utf-8", 
                                         on_bad_lines="skip"):
                    chunk = self._validate_chunk(chunk, drop_missing)
                    chunks.append(chunk)
                self.df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            else:
                self._validate_missing_values(drop_missing)
                self._validate_data_types()
            logger.info({"message": "All validations completed successfully", "row_count": len(self.df)})
            return self.df
        except Exception as e:
            logger.error({"message": "Data validation failed", "error": str(e)})
            raise CustomException("Data validation failed", str(e))

    def _validate_chunk(self, chunk: pd.DataFrame, drop_missing: bool) -> pd.DataFrame:
        """
        Validate a single chunk of the DataFrame.
        """
        chunk = chunk.copy()
        self._validate_missing_values(chunk, drop_missing)
        self._validate_data_types(chunk)
        return chunk

    def _validate_columns(self):
        """
        Check for required columns in the DataFrame.
        """
        missing_required = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_required:
            logger.error({"message": "Missing required columns", "missing_columns": missing_required})
            raise CustomException("Missing required columns", f"Missing: {missing_required}")
        logger.info({"message": "Required columns validated", "columns": self.REQUIRED_COLUMNS})
        available_optional = [col for col in self.OPTIONAL_COLUMNS if col in self.df.columns]
        logger.debug({"message": "Optional columns detected", "columns": available_optional})

    def _validate_missing_values(self, df: pd.DataFrame, drop_missing=True):
        """
        Check for missing or empty values in required columns.
        """
        invalid_rows = df[self.REQUIRED_COLUMNS].isnull().any(axis=1) | \
                       (df[self.REQUIRED_COLUMNS].apply(lambda x: x.astype(str).str.strip() == '')).any(axis=1)
        num_invalid = invalid_rows.sum()
        if num_invalid > 0:
            logger.warning({"message": "Found invalid rows in required columns", "count": num_invalid})
            if drop_missing:
                df = df[~invalid_rows].reset_index(drop=True)
                logger.info({"message": "Dropped invalid rows", "dropped_count": num_invalid})
        else:
            logger.info({"message": "No missing or empty values in required columns"})
        self.df = df

    def _validate_data_types(self, df: pd.DataFrame = None):
        """
        Convert columns to string and strip whitespace.
        """
        df = df if df is not None else self.df
        for col in self.REQUIRED_COLUMNS + [c for c in self.OPTIONAL_COLUMNS if c in df.columns]:
            df[col] = df[col].astype(str).str.strip()
        logger.debug({"message": "Data types validated and converted to string"})
        if df is not self.df:
            return df
