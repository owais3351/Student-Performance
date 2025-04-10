import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    logging.info("stores the file path for preprocessor.pkl")
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformation_object(self):
        """
        This funcation is responsible for data transformation

        """
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            logging.info('Handling missing values using simpleImputer for numerical columns')
            logging.info("perform standardization for numerical columns")
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info('handling missing value, encoding,standardization for categorical features')   
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('One_hot_encoder',OneHotEncoder(handle_unknown="ignore")),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'categorical columns : {categorical_columns}')
            logging.info(f'numerical columns : {numerical_columns}')

            #combine num_pipeline and cat_pipeline using column transformer
            processor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns),
                ]
            )

            return processor


        except Exception as e:
            raise CustomException(e,sys)
        
    
    logging.info('starting with data transformation')
    logging.info('This funcation reads train and test CSV file')
    logging.info('Calls get_data_tramsformation_object to get the preprocessing_obj')

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read Train and Test data completed')

            logging.info('obatining preprocessor object')
            preprocessor_obj=self.get_data_transformation_object()
            
            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            # Choose independent and dependent features for train_df 
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            # Choose independent and dependent features for test_df 
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f'Applying preprocessing object on training ddataframe anf testing dataframe')
            
            # Fit and transform
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            # concatanate/combine input features

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            # Store preprocessor_obj as pkl file in save_object
            # Save_object funcation in defined in utils.py

            logging.info(f'save preprocessing object')
            logging.info('call save_object funcation form utils.py')
            

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,

                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )




        except Exception as e:
            raise CustomException(e,sys)
            