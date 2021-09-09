import os
from Scripts import read_data as rd
import streamlit as st 


base = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base, "data/raw/lstm_1_Result.csv")
dataframe = rd.preprocess_data(rd.read_dataset(path))

dataframe = dataframe.drop(['Actual','Error'],axis = 1)

st.write('''
 Share Market Forecasting 
 ''')
choose = 'DADBL'
st.write("Please choose A for ADBL bank")
option = st.selectbox('Please Choose option',(choose))

st.write()
if option ==  'A':
    st.line_chart(dataframe.tail(5))
