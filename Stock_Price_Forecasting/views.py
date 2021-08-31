import os
import pickle
import streamlit as st
import joblib
from Stock_Price_Prediction.src.utils import split_dataset
from Scripts import read_data as rd

base = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    
    pickled_model = pickle.load(open('model_checkpoints/model.pickle', 'rb'))

