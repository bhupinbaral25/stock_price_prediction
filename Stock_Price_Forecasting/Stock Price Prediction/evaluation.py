from sklearn.preprocessing import MinMaxScaler
import math
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def inverse_scalar(dataframe: np.ndarray) -> np.ndarray:
    """ """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.inverse_transform(dataframe)


def inverse_scaling(model, n_lag, X_test, Y_test):
    """ """
    y_predict = model.predict(X_test)
    test_X = X_test.reshape((X_test.shape[0], n_lag * X_test.shape(2)))

    inverse_ypredict = inverse_scalar(
        np.concatenate((y_predict, test_X[:, -(X_test.shape(2) - 1) :]), axis=1)
    )[:, 0]
    inverse_ytrue = inverse_scalar(
        np.concatenate(
            (Y_test.reshape((len(Y_test), 1)), test_X[:, -(X_test.shape(2) - 1) :]),
            axis=1,
        )
    )[:, 0]

    return inverse_ypredict, inverse_ytrue


def evaluateModel(y_actual : np.ndarray, y_predicted : np.ndarray) -> dict:
    """ """

    result_dict = {
        "Test RMSE:": math.sqrt(mean_squared_error(y_actual, y_predicted)),
        "R2_Score: ": r2_score(y_actual, y_predicted),
        "MAE:": mean_absolute_error(y_actual, y_predicted),
    }

    return result_dict
