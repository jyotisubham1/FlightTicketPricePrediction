import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.logger import logging
from src.exception import CustomException

def main():
    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        X_train, X_test = data_transformation.transform_data(train_data_path, test_data_path)

        # Model Training
        training_pipeline = TrainingPipeline()
        model = training_pipeline.train(X_train, y_train)  # Assuming y_train is available

        # Model Prediction
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(X_test)

        # Print Regression Scores
        # Assuming you have regression scores to print
        logging.info('Regression scores:')
        print_regression_scores(predictions, y_test)  # Implement print_regression_scores function

    except CustomException as e:
        logging.error(f'CustomException: {e}')
        sys.exit(1)

if __name__ == "__main__":
    main()
