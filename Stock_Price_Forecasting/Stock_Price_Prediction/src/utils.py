import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def series_to_supervised(dataframe: np.ndarray, n_lag: int = 5, num_label: int = 1, dropnan: bool = True) -> pd.DataFrame:
    """ """
    column_size = dataframe.shape[1]
    dataframe = pd.DataFrame(dataframe)
    columns, names = [], []

    for index in range(n_lag, 0, -1):
        columns.append(dataframe.shift(index))
        names += [
            ("var%d(t-%d)" % (coln_name + 1, index))
            for coln_name in range(column_size)
        ]

    for index in range(num_label):
        columns.append(dataframe.shift( - index))
        if index == 0:
            names += [
                ("var%d(t)" % (coln_name + 1)) for coln_name in range(column_size)
            ]
        else:
            names += [
                ("var%d(t+%d)" % (coln_name + 1, index))
                for coln_name in range(column_size)
            ]       
    new_dataframe = pd.concat(columns, axis = 1)
    new_dataframe.columns = names
    if dropnan:
        new_dataframe.dropna(inplace = True)

    return new_dataframe

def scale_dataset(dataframe : pd.DataFrame) -> np.ndarray:
    """ """
    values = dataframe.values
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    return scaler.fit_transform(values)


   
# def __init__(self, dataframe: pd.DataFrame) -> None:
#     self.dataframe = dataframe


def split_dataset(
    dataframe, split_size: float = 0.8, n_lag: int = 1, reshape3D: bool = True ) -> np.ndarray:
    """
    reshape input to [samples, time steps, features]
    """
    new_data_set = series_to_supervised(scale_dataset(dataframe),n_lag=12)
    data_set = new_data_set.values
    # split the data into trainset, validationset, testset
    training_size = math.ceil(len(dataframe) * split_size)
    train_size = math.ceil(training_size * split_size)
    train_set = data_set[:train_size, :]
    validation_set = data_set[train_size:training_size, :]
    test_set = data_set[training_size:, :]
    # split the different dataset into xtrain,xtest,ytrain,ytest,xvalidation,yvalidation
    X_train, Y_train = (
        train_set[:, : n_lag * dataframe.shape[1]],
        train_set[:, - dataframe.shape[1]],
    )
    X_validation, Y_validation = (
        validation_set[:, :n_lag * dataframe.shape[1]],
        validation_set[:, - dataframe.shape[1]],
    )
    X_test, Y_test = (
        test_set[:, : n_lag * dataframe.shape[1]],
        test_set[:, - dataframe.shape[1]],
    )
    # Reshape the data X value into 3D
    if reshape3D:
        X_train = np.reshape(
            X_train, (X_train.shape[0], n_lag, dataframe.shape[1])
        )
        X_validation = np.reshape(
            X_validation, (X_validation.shape[0], n_lag, dataframe.shape[1])
        )
        X_test = np.reshape(
            X_test, (X_test.shape[0], n_lag, dataframe.shape[1])
        )
        #Y_test = np.reshape(X_test, (len(Y_test), 1))

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

  