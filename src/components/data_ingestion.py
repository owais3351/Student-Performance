# import basic libraries-->to use customException and Logging
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split

# Dataclasses module in python provide @dataclass decorator which simplifies the cration of class that primarily store data:
from dataclasses import dataclass

@dataclass
class DataIngestionConfig: ## class store file path
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion: ## This class is responsible for loading the data before transforming and training ML models:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() ## created an instance of DataIngestionConfig() and assign it to "self.ingestion_config".
    
    # initiate_data_ingestion is actually a method defined for DataIngestion Class:

    def initiate_data_ingestion(self): ## This Funcation is responsible for reading  data from a CSV file,splitting,saving the split datasets into predefined locations.
        logging.info("Starts the data ingestion method")

        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Read the dataset from local source")
            
            # creates a directory where the train,test and raw data files will be stored
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Saves the raw dataset(df) as  a CSV file in the location given by "(self.ingestion_config.raw_data_path"
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()