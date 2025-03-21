import os 
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):

    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
## define another funcation that evaluate model performance:

def evaluate_models(x_train,y_train,x_test,y_test,models):
    
    try:

        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            # model Training
            model.fit(x_train,y_train)

            #predictions for training data
            y_pred_train=model.predict(x_train)

            # predictions for test data
            y_pred_test=model.predict(x_test)

            #model performance evaluation:
            train_model_score=r2_score(y_train,y_pred_train)
            test_model_score=r2_score(y_test,y_pred_test)

            report[list(models.keys())[i]]=test_model_score

            logging.info('get all model names(keys) from the modell dictionary :  model.keys()')
            logging.info('convert the keys in the list and select the i-th model name : list(model.keys())[i]')
            logging.info('store the r2_Score of the i-th model on the test dataset : test_model_score')
            logging.info("saves the model's test test score in the dictionay : report[------]=test_model_score")

        return report

    except Exception as e:
        raise CustomException(e,sys)