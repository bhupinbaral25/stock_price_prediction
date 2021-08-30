import os 
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature
from Stock_Price_Prediction import train ,evaluation
from Stock_Price_Prediction.src.utils import split_dataset, series_to_supervised,scale_dataset
from Scripts import read_data as rd


base = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    n_lag = 12
    path = os.path.join(base, "data/raw/Agricultural_development_bank_limited.xlsx")
    dataframe = rd.preprocess_data(rd.read_dataset(path, file_type='xlsx'))
    
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = split_dataset(
        dataframe, n_lag=n_lag
    )
    hyper_parameters  = {
        'X_train': X_train,
        'Y_train': Y_train,
        'epochs': 100,
        'batch_size' : 64,
        'validation_data' : (X_validation, Y_validation),
        'verbose' : 1,
        'shuffle': False,
        'units' : 50
    }

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

        (rmse, mae, r2) = evaluation.evaluate_model(LSTM_model,n_lag,X_test,Y_test)

        
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            signature = infer_signature(X_test, LSTM_model.predict(X_test))
            mlflow.keras.log_model(LSTM_model, "LSTM_time_series_foreccasting", signature = signature)
