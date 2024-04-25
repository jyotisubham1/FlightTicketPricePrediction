from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.logger import logging
from src.exception import CustomException
from src.pipelines.model_trainer import preprocess_pipeline
import sys

class TrainingPipeline:
    def __init__(self):
        self.models = {
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'SVR': SVR(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'LinearRegression': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(),
            'AdaBoostRegressor': AdaBoostRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor()
        }
        
    def train(self, X_train, y_train):
        try:
            logging.info('Training pipeline started')
            trained_models = {}
            
            for name, model in self.models.items():
                model_pipeline = Pipeline([
                    ('preprocessor', preprocess_pipeline),
                    ('model', model)
                ])
                model_pipeline.fit(X_train, y_train)
                trained_models[name] = model_pipeline
                
                # Predict on the training data
                y_pred_train = model_pipeline.predict(X_train)
                
                # Calculate regression metrics
                r2_train = r2_score(y_train, y_pred_train)
                mae_train = mean_absolute_error(y_train, y_pred_train)
                rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
                
                # Print or log the regression metrics
                logging.info(f'Model: {name}')
                logging.info(f'R2 score for train data: {r2_train}')
                logging.info(f'Mean absolute error: {mae_train}')
                logging.info(f'Root mean squared error: {rmse_train}')
                logging.info('=' * 75)
            
            logging.info('Training pipeline completed')
            
            return trained_models
        
        except Exception as e:
            logging.error('Error occurred during training')
            raise CustomException(e, sys)
