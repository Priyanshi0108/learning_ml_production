from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (
    CustomData,
    PredictPipeline
)

from src.logger import logger


app = Flask(__name__)


@app.route("/")
def home():

    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict_datapoints():

    try:

        # ==========================================
        # Get data from HTML form
        # ==========================================

        data = CustomData(

            gender=request.form.get(
                "gender"
            ),

            race_ethnicity=request.form.get(
                "race_ethnicity"
            ),

            parental_level_of_education=request.form.get(
                "parental_level_of_education"
            ),

            lunch=request.form.get(
                "lunch"
            ),

            test_preparation_course=request.form.get(
                "test_preparation_course"
            ),

            reading_score=float(
                request.form.get(
                    "reading_score"
                )
            ),

            writing_score=float(
                request.form.get(
                    "writing_score"
                )
            )
        )

        # ==========================================
        # Convert input → DataFrame
        # ==========================================

        final_data = (
            data.get_data_as_dataframe()
        )

        logger.info(
            "Input data converted to DataFrame"
        )

        # ==========================================
        # Prediction Pipeline
        # ==========================================

        predict_pipeline = PredictPipeline()

        prediction = (
            predict_pipeline.predict(
                final_data
            )
        )

        # ==========================================
        # Load Model Metrics
        # ==========================================

        metrics = (
            predict_pipeline.get_model_metrics()
        )

        # ==========================================
        # Final Prediction
        # ==========================================

        result = round(
            float(prediction[0]),
            2
        )

        logger.info(
            f"Final prediction: {result}"
        )

        return render_template(
            "result.html",
            prediction=result,
            metrics=metrics
        )

    except Exception as e:

        logger.exception(
            "Prediction request failed"
        )

        return render_template(
            "result.html",
            error="Unable to generate prediction."
        )


if __name__ == "__main__":

    app.run(
        debug=True
    )