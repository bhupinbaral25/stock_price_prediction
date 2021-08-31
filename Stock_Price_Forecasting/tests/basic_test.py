import pytest
import pandas as pd
from Scripts import read_data as rd
#from Stock_Price_Prediction import evaluation as evl
path = '../data/raw/Agricultural_development_bank_limited.xlsx'

dataset = pd.read_excel(path)


