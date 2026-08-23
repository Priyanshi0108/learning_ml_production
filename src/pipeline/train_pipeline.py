from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



DataIngestion().initiate_data_ingestion()

DataValidation().initiate_data_validation()

train_arr,test_arr,_ = DataTransformation().initiate_data_transformation()


best_model, score = ModelTrainer().initiate_model_training(
    train_arr ,
    test_arr
)

print(best_model,score)