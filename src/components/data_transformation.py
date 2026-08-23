import os
import pickle
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logger
from src.exception import DataTransformationError


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.config = DataTransformationConfig()

    def get_preprocessor_object(self):

        try:
            logger.info("Creating preprocessing pipeline")

            numerical_columns = [
                "reading_score",
                "writing_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("numerical", num_pipeline, numerical_columns),
                ("categorical", cat_pipeline, categorical_columns)
            ])

            logger.info("Preprocessor created successfully")

            return preprocessor

        except Exception as e:
            raise DataTransformationError(
                "Failed to create preprocessing pipeline",
                e
            )

    def initiate_data_transformation(self):

        logger.info("Starting data transformation")

        try:
            train_df = pd.read_csv(
                os.path.join("artifacts", "train.csv")
            )

            test_df = pd.read_csv(
                os.path.join("artifacts", "test.csv")
            )

            target_column = "math_score"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_preprocessor_object()

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]

            with open(
                self.config.preprocessor_path,
                "wb"
            ) as file:
                pickle.dump(preprocessor, file)

            logger.info("Preprocessor saved successfully")

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_path
            )

        except Exception as e:
            raise DataTransformationError(
                "Data transformation failed",
                e
            )