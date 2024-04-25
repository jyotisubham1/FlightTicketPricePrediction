from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from src.logger import logging
from src.exception import CustomException
from src.pipelines.model_trainer import preprocess_pipeline

class PredictionPipeline:
    def __init__(self):
        self.model = RandomForestRegressor()
        
    def predict(self, X):
        try:
            model_pipeline = Pipeline([
                ('preprocessor', preprocess_pipeline),
                ('model', self.model)
            ])
            
            logging.info('Prediction pipeline started')
            predictions = model_pipeline.predict(X)
            logging.info('Prediction pipeline completed')
            
            return predictions
        
        except Exception as e:
            logging.info('Error occurred during prediction')
            raise CustomException(e, sys)
