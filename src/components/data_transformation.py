import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.logger import logging
from src.exception import CustomException

class DataTransformation:
    def __init__(self):
        self.numeric_features = ['NumericFeature1', 'NumericFeature2']
        self.categorical_features = ['CategoricalFeature1', 'CategoricalFeature2']
        
    def transform_data(self, X_train, X_test):
        try:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.numeric_features),
                    ('cat', categorical_transformer, self.categorical_features)])
            
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            logging.info('Data transformation completed')
            
            return X_train_transformed, X_test_transformed
        
        except Exception as e:
            logging.info('Error occurred during data transformation')
            raise CustomException(e, sys)
