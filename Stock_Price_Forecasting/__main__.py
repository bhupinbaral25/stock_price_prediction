import os 
import mlflow
import pickle
import keras
import mlflow.keras
from mlflow.models.signature import infer_signature
from numpy.lib import utils
from Stock_Price_Prediction import train ,evaluation
from Stock_Price_Prediction.src.utils import split_dataset, scale_dataset
from Scripts import read_data as rd

base = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    n_lag = 12
    path = os.path.join(base, "data/processed/final_dataset.csv")
    dataframe = rd.preprocess_data(rd.read_dataset(path))

    '''Spliting the dataset and reshape them into the compatible input for model'''
    
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = split_dataset(
    dataframe, n_lag = n_lag)
    
    ''' hyper parameter for fine tunning for the model '''

    hyper_parameters  = {
        'X_train': X_train,
        'Y_train': Y_train,
        'epochs': 100,
        'batch_size' : 64,
        'validation_data' : (X_validation, Y_validation),
        'verbose' : 1,
        'shuffle': False,
        'units' : 100
    }
    '''Model Building and storing using Ml flow'''
    with mlflow.start_run():
        LSTM_model = train.built_model(
        X_train = X_train,
        units = hyper_parameters['units']
        )
        LSTM_model.fit(
            hyper_parameters['X_train'],
            hyper_parameters['Y_train'],
            epochs = hyper_parameters['epochs'],
            batch_size = hyper_parameters['batch_size'],
            validation_data = hyper_parameters['validation_data'],
            verbose = hyper_parameters['verbose'],
            shuffle = hyper_parameters['shuffle']
        )

        predicted_qualities = LSTM_model.predict(X_test)
        
        LSTM_model.save('model_checkpoint')
        
        (rmse, mae, r2) = evaluation.evaluate_model(predicted_qualities, Y_test)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
       
        signature = infer_signature(X_test, LSTM_model.predict(X_test))
        mlflow.keras.log_model(LSTM_model, "LSTM_time_series_foreccasting", signature = signature)
        mlflow.keras.log_model(LSTM_model, "model")

    
