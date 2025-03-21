import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:  ## stores file path of folder where our pickle file is saved:
    trained_model_file_path=os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.modle_trainer_congig=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array): ## Train_arr,test_arr are actually output of the data_transformation.py

        try:
            logging.info("split training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[: , : -1], # x_train
                train_array[: , -1], # y_train
                test_array[: , : -1], # x_test
                test_array[: , -1] # y_test
            )

            logging.info("choose different models for predictions")

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGB Reh=gressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            ## Evaluate Model Performance 
            ## Use evaluate_modles funcation-->is defined in utils.py
            ## Model_report funcation will store r2_score for all models in form of dict
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            # To get the best model score from dict:
            best_model_score=max(sorted(model_report.values()))

            # To get best model name from dict:
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # to get best model with high r2_score
            best_model=models[best_model_name]


            if best_model_score < 0.6:
                raise CustomException('No best model found')
            
            logging.info(f"Best found model on both training and testing dataset")


            save_object(
                file_path=self.modle_trainer_congig.trained_model_file_path,
                obj=best_model
            )

            # prediction for best model:

            predicted=best_model.predict(x_test)

            r2_square=r2_score(y_test,predicted)

            return r2_square
        
        
        except Exception as e:
            raise CustomException(e,sys)

