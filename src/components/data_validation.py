import os
from dataclasses import dataclass

import pandas as pd

from src.logger import logger
from src.exception import DataValidationError


@dataclass
class DataValidationConfig:
    validation_status_path = os.path.join("artifacts", "validation_status.txt")


class DataValidation:

    def __init__(self):
        self.config = DataValidationConfig()

    def initiate_data_validation(self):

        logger.info("Data Validation component started")

        try:
            train_df = pd.read_csv(os.path.join("artifacts", "train.csv"))
            test_df = pd.read_csv(os.path.join("artifacts", "test.csv"))

            required_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                "reading_score",
                "writing_score",
                "math_score"
            ]

            missing = set(required_columns) - set(train_df.columns)

            if missing:
                raise DataValidationError(f"Missing columns: {missing}")

            if train_df.empty:
                raise DataValidationError("Train dataset is empty")

            if test_df.empty:
                raise DataValidationError("Test dataset is empty")

            duplicates = train_df.duplicated().sum()

            logger.info(f"Duplicate rows: {duplicates}")

            with open(self.config.validation_status_path, "w") as f:
                f.write("Validation Passed")

            logger.info("Data validation completed")

        except Exception as e:
            raise DataValidationError("Data validation failed", e)