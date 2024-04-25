from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from src.logger import logging
from src.exception import CustomException

# Define preprocess_pipeline if needed
preprocess_pipeline = None

def train_individual_models(X_train, y_train):
    try:
        dt = DecisionTreeRegressor()
        svr = SVR()
        knn = KNeighborsRegressor()
        lr = LinearRegression()
        
        models = {'DecisionTreeRegressor': dt, 'SVR': svr, 'KNeighborsRegressor': knn, 'LinearRegression': lr}
        trained_models = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        logging.info('Individual models trained successfully')
        return trained_models
    
    except Exception as e:
        logging.info('Error occurred during individual model training')
        raise CustomException(e, sys)

def train_ensemble_models(X_train, y_train):
    try:
        rfr = RandomForestRegressor()
        ad = AdaBoostRegressor()
        gd = GradientBoostingRegressor()
        
        models = {'RandomForestRegressor': rfr, 'AdaBoostRegressor': ad, 'GradientBoostingRegressor': gd}
        trained_models = {}
        
        for name, model in models.items():
            model_pipeline = Pipeline([
                ('preprocessor', preprocess_pipeline),
                ('model', model)
            ])
            model_pipeline.fit(X_train, y_train)
            trained_models[name] = model_pipeline
        
        logging.info('Ensemble models trained successfully')
        return trained_models
    
    except Exception as e:
        logging.info('Error occurred during ensemble model training')
        raise CustomException(e, sys)
