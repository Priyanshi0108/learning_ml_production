import os
import sys
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import DataIngestionError
from src.logger import logger


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw.csv")


class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Entered the data ingestion component")        
        try:
            df = pd.read_csv("/Users/rajpoot/Desktop/learningML/notebook/data/stud.csv")
            # df = pd.read_csv(os.path.join("notebook","data","stud.csv"))
            logger.info("Dataset loaded successfully")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index= False,
                header = True
            )

            logger.info("Raw data saved")

            train_set , test_set = train_test_split(
                df,test_size=0.2,random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                 self.ingestion_config.test_data_path,
                 index=False,
                 header=True
            )

            logger.info("Train and test data are saved successfully ")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except DataIngestionConfig as e:
            logger.exception("Error occurred during data ingestion")
            raise DataIngestionError("Failed during data ingestion.",e)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path , test_path = obj.initiate_data_ingestion()

    print(train_path)
    print(test_path)
        