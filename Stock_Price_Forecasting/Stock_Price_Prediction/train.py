from keras.backend import dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np


def built_model(X_train: np.ndarray, units: int = 120, dropout: int = 0.2):
    """
    This  functions initialize the model with neccessary parameters X_train shaped
    is used to identify the input shape of the model.
    """
    model = Sequential()
    model.add(
        LSTM(
            units = units,
            return_sequences = True,
            dropout = dropout,
            input_shape = (X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(LSTM(units = units, return_sequences = True))
    model.add(LSTM(units = units))
    model.add(Dense(1))
    model.compile(loss = "mse", optimizer = "adam", metrics = ["mse", "mae"])

    return model
