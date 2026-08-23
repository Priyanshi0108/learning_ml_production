from src.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)


data = CustomData(
    gender="female",
    race_ethnicity="group C",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=72,
    writing_score=74
)

df = data.get_data_as_dataframe()

print(df)

pipeline = PredictPipeline()

prediction = pipeline.predict(df)

print("Predicted Math Score:", prediction[0])