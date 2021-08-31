from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def inverse_scalar(dataframe: np.ndarray) -> np.ndarray:
    """ """
    scaler = MinMaxScaler(feature_range=(-1, 1))

    return scaler.inverse_transform(dataframe)


def inverse_scaling(
    model, n_lag: int, X_test: np.ndarray, Y_test: np.ndarray
) -> np.ndarray:
    """This functions inverse the scaling of model output and lable value"""
    y_predict = model.predict(X_test)
    X_test = np.array(X_test.reshape((X_test.shape[0], n_lag * X_test.shape(2))))
    inverse_ypredict = inverse_scalar(
        np.concatenate((y_predict, X_test[:, -(X_test.shape(2) - 1) :]), axis=1)
    )[:, 0]
    inverse_ytrue = inverse_scalar(
        np.concatenate(
            (Y_test.reshape((len(Y_test), 1)), X_test[:, -(X_test.shape(2) - 1) :]),
            axis=1,
        )
    )[:, 0]

    return inverse_ypredict, inverse_ytrue


def evaluate_model(predicted: np.ndarray, actual: np.ndarray) -> float:
    """This functions returns the value of different evaluation metrics"""

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return rmse, mae, r2
