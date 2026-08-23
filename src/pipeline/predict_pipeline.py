import os
import pandas as pd

from src.exception import MLProjectError
from src.logger import logger
from src.utils import load_object


class PredictPipeline:

    def predict(self, features):

        try:
            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            logger.info("Loading model")

            model = load_object(model_path)

            logger.info("Loading preprocessor")

            preprocessor = load_object(preprocessor_path)

            transformed = preprocessor.transform(features)

            prediction = model.predict(transformed)

            return prediction

        except Exception as e:
            raise MLProjectError(
                "Prediction pipeline failed",
                e
            )


class CustomData:

    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        test_preparation_course,
        reading_score,
        writing_score
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):

        try:

            data = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [
                    self.parental_level_of_education
                ],
                "lunch": [self.lunch],
                "test_preparation_course": [
                    self.test_preparation_course
                ],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            logger.info(
                "Custom input converted to DataFrame"
            )

            return pd.DataFrame(data)

        except Exception as e:
            raise MLProjectError(
                "Failed to create DataFrame",
                e
            )