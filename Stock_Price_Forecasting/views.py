import keras
import matplotlib.pyplot as plt
import os
from Stock_Price_Prediction.src.utils import split_dataset, series_to_supervised
from Scripts import read_data as rd
import streamlit as st 


model = keras.models.load_model('model_checkpoint')
n_lag = 12
base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, "data/processed/final_dataset.csv")
dataframe = rd.preprocess_data(rd.read_dataset(path))

'''Spliting the dataset and reshape them into the compatible input for model'''
    
X_train, X_validation, X_test, Y_train, Y_validation, Y_test = split_dataset(
    dataframe, n_lag = n_lag)

st.text('Hello User Here is the forcasting for ADBl ')

predict = model.predict(X_test)
actual = Y_test
st.plotly_chart(predict)








