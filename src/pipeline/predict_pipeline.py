import os
import pandas as pd

from src.exception import MLProjectError
from src.logger import logger
from src.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            logger.info("Prediction pipeline started")

            # ==========================================
            # Model Path
            # ==========================================

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            # ==========================================
            # Preprocessor Path
            # ==========================================

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            logger.info(
                f"Loading model from: {model_path}"
            )

            model = load_object(model_path)

            logger.info(
                f"Loading preprocessor from: "
                f"{preprocessor_path}"
            )

            preprocessor = load_object(
                preprocessor_path
            )

            # ==========================================
            # Transform Input
            # ==========================================

            logger.info(
                "Transforming prediction input"
            )

            transformed_data = (
                preprocessor.transform(features)
            )

            # ==========================================
            # Prediction
            # ==========================================

            logger.info(
                "Generating prediction"
            )

            prediction = model.predict(
                transformed_data
            )

            logger.info(
                f"Prediction generated: {prediction}"
            )

            return prediction

        except Exception as e:

            raise MLProjectError(
                "Prediction pipeline failed",
                e
            )

    # ==============================================
    # Load Model Metrics
    # ==============================================

    def get_model_metrics(self):

        try:

            metrics_path = os.path.join(
                "artifacts",
                "model_metrics.pkl"
            )

            logger.info(
                "Loading model metrics"
            )

            metrics = load_object(
                metrics_path
            )

            return metrics

        except Exception as e:

            raise MLProjectError(
                "Failed to load model metrics",
                e
            )


# ==================================================
# Custom Data
# ==================================================

class CustomData:

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float
    ):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = (
            parental_level_of_education
        )

        self.lunch = lunch

        self.test_preparation_course = (
            test_preparation_course
        )

        self.reading_score = reading_score

        self.writing_score = writing_score


    def get_data_as_dataframe(self):

        try:

            logger.info(
                "Converting custom input "
                "into DataFrame"
            )

            data = {

                "gender": [
                    self.gender
                ],

                "race_ethnicity": [
                    self.race_ethnicity
                ],

                "parental_level_of_education": [
                    self.parental_level_of_education
                ],

                "lunch": [
                    self.lunch
                ],

                "test_preparation_course": [
                    self.test_preparation_course
                ],

                "reading_score": [
                    self.reading_score
                ],

                "writing_score": [
                    self.writing_score
                ]
            }

            dataframe = pd.DataFrame(data)

            logger.info(
                f"Custom DataFrame created: "
                f"{dataframe.shape}"
            )

            return dataframe

        except Exception as e:

            raise MLProjectError(
                "Failed to create prediction DataFrame",
                e
            )