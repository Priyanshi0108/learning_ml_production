import os
import pickle

from dataclasses import dataclass

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.logger import logger
from src.exception import ModelTrainingError


# ==========================================================
# Model Trainer Configuration
# ==========================================================

@dataclass
class ModelTrainerConfig:

    model_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )

    metrics_path: str = os.path.join(
        "artifacts",
        "model_metrics.pkl"
    )


# ==========================================================
# Model Trainer
# ==========================================================

class ModelTrainer:

    def __init__(self):

        self.config = ModelTrainerConfig()


    def initiate_model_training(
        self,
        train_arr,
        test_arr
    ):

        logger.info(
            "========== Model Training Started =========="
        )

        try:

            # ==================================================
            # 1. Separate Features and Target
            # ==================================================

            logger.info(
                "Separating features and target"
            )

            X_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logger.info(
                f"X_train shape: {X_train.shape}"
            )

            logger.info(
                f"y_train shape: {y_train.shape}"
            )

            logger.info(
                f"X_test shape: {X_test.shape}"
            )

            logger.info(
                f"y_test shape: {y_test.shape}"
            )


            # ==================================================
            # 2. Define Models
            # ==================================================

            models = {

                "XGBoost": XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                ),

                "CatBoost": CatBoostRegressor(
                    loss_function="RMSE",
                    random_state=42,
                    verbose=0,
                    thread_count=-1
                )
            }


            # ==================================================
            # 3. Define Hyperparameter Search Space
            # ==================================================

            params = {

                "XGBoost": {

                    "n_estimators": [
                        100,
                        200
                    ],

                    "max_depth": [
                        3,
                        5
                    ],

                    "learning_rate": [
                        0.01,
                        0.1
                    ],

                    "subsample": [
                        0.8,
                        1.0
                    ],

                    "colsample_bytree": [
                        0.8,
                        1.0
                    ]
                },


                "CatBoost": {

                    "iterations": [
                        100,
                        200
                    ],

                    "depth": [
                        4,
                        6
                    ],

                    "learning_rate": [
                        0.01,
                        0.1
                    ],

                    "l2_leaf_reg": [
                        1,
                        3
                    ]
                }
            }


            # ==================================================
            # 4. Store Results
            # ==================================================

            model_results = {}

            best_model = None
            best_model_name = None
            best_cv_score = float("-inf")
            best_params = None


            # ==================================================
            # 5. GridSearchCV for Each Model
            # ==================================================

            for name, model in models.items():

                logger.info(
                    f"========== Training {name} =========="
                )


                grid_search = GridSearchCV(

                    estimator=model,

                    param_grid=params[name],

                    cv=3,

                    scoring="r2",

                    n_jobs=-1,

                    verbose=1,

                    return_train_score=False
                )


                # ==================================================
                # 6. Train Model + Hyperparameter Tuning
                # ==================================================

                logger.info(
                    f"Starting GridSearchCV for {name}"
                )

                grid_search.fit(
                    X_train,
                    y_train
                )

                logger.info(
                    f"GridSearchCV completed for {name}"
                )


                # ==================================================
                # 7. Get Best Parameters
                # ==================================================

                best_estimator = (
                    grid_search.best_estimator_
                )

                best_params_for_model = (
                    grid_search.best_params_
                )

                cv_score = (
                    grid_search.best_score_
                )


                logger.info(
                    f"{name} Best CV R2: "
                    f"{cv_score:.4f}"
                )

                logger.info(
                    f"{name} Best Parameters: "
                    f"{best_params_for_model}"
                )


                # ==================================================
                # 8. Store Model Results
                # ==================================================

                model_results[name] = {

                    "cv_r2_score": float(
                        cv_score
                    ),

                    "best_params":
                        best_params_for_model
                }


                # ==================================================
                # 9. Select Best Model Using CV Score
                # ==================================================

                if cv_score > best_cv_score:

                    best_cv_score = cv_score

                    best_model = best_estimator

                    best_model_name = name

                    best_params = (
                        best_params_for_model
                    )


            # ==================================================
            # 10. Safety Check
            # ==================================================

            if best_model is None:

                raise ModelTrainingError(
                    "No model was successfully trained."
                )


            logger.info(
                f"Best model selected using CV: "
                f"{best_model_name}"
            )


            # ==================================================
            # 11. Final Evaluation on Test Set
            # ==================================================

            logger.info(
                "Evaluating final model on test set"
            )

            test_predictions = (
                best_model.predict(X_test)
            )


            # -----------------------------
            # R2 Score
            # -----------------------------

            test_r2 = r2_score(
                y_test,
                test_predictions
            )


            # -----------------------------
            # MAE
            # -----------------------------

            test_mae = mean_absolute_error(
                y_test,
                test_predictions
            )


            # -----------------------------
            # RMSE
            # -----------------------------

            test_rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    test_predictions
                )
            )


            logger.info(
                f"Final Model: {best_model_name}"
            )

            logger.info(
                f"Test R2: {test_r2:.4f}"
            )

            logger.info(
                f"Test MAE: {test_mae:.4f}"
            )

            logger.info(
                f"Test RMSE: {test_rmse:.4f}"
            )


            # ==================================================
            # 12. Add Final Test Metrics
            # ==================================================

            model_results[
                best_model_name
            ][
                "test_r2_score"
            ] = float(test_r2)

            model_results[
                best_model_name
            ][
                "test_mae"
            ] = float(test_mae)

            model_results[
                best_model_name
            ][
                "test_rmse"
            ] = float(test_rmse)


            # ==================================================
            # 13. Create Final Metrics Dictionary
            # ==================================================

            metrics = {

                "best_model":
                    best_model_name,

                "cv_r2_score":
                    float(best_cv_score),

                "test_r2_score":
                    float(test_r2),

                "test_mae":
                    float(test_mae),

                "test_rmse":
                    float(test_rmse),

                "best_params":
                    best_params,

                "model_comparison":
                    model_results
            }


            # ==================================================
            # 14. Create Artifacts Directory
            # ==================================================

            artifacts_dir = os.path.dirname(
                self.config.model_path
            )

            os.makedirs(
                artifacts_dir,
                exist_ok=True
            )


            # ==================================================
            # 15. Save Best Model
            # ==================================================

            logger.info(
                "Saving best model"
            )

            with open(
                self.config.model_path,
                "wb"
            ) as file:

                pickle.dump(
                    best_model,
                    file
                )


            logger.info(
                f"Model saved successfully: "
                f"{self.config.model_path}"
            )


            # ==================================================
            # 16. Save Model Metrics
            # ==================================================

            logger.info(
                "Saving model metrics"
            )

            with open(
                self.config.metrics_path,
                "wb"
            ) as file:

                pickle.dump(
                    metrics,
                    file
                )


            logger.info(
                f"Metrics saved successfully: "
                f"{self.config.metrics_path}"
            )


            # ==================================================
            # 17. Final Summary
            # ==================================================

            logger.info(
                "=========================================="
            )

            logger.info(
                "MODEL TRAINING COMPLETED"
            )

            logger.info(
                f"Best Model: {best_model_name}"
            )

            logger.info(
                f"CV R2: {best_cv_score:.4f}"
            )

            logger.info(
                f"Test R2: {test_r2:.4f}"
            )

            logger.info(
                f"Test MAE: {test_mae:.4f}"
            )

            logger.info(
                f"Test RMSE: {test_rmse:.4f}"
            )

            logger.info(
                f"Best Parameters: {best_params}"
            )

            logger.info(
                "=========================================="
            )


            # ==================================================
            # 18. Return
            # ==================================================

            return (
                best_model_name,
                metrics
            )


        except Exception as e:

            logger.exception(
                "Error occurred during model training"
            )

            raise ModelTrainingError(
                "Model training failed",
                e
            )