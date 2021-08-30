import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

def figure_plot(
    dataframe: pd.DataFrame, x_axis: str, y_axis: str, choose_plot: str) -> plt.Figure:
    '''
    chooseplot = 'matplot' for matplotlib
                 'plotly' for plotly
    '''
    if choose_plot == "matplot":
        plt.figure(figsize=(20, 12))
        figure = plt.plot(dataframe[x_axis], dataframe[y_axis])

    elif choose_plot == "plotly":
        figure = px.line(dataframe, x = x_axis, y = y_axis)

    return figure


def Check_seasonality(dataframe: pd.DataFrame) -> None:
    '''
    Multiplicative Decompostion for checking seasonality
    '''
    for column in dataframe.columns:
        mul_decomposition = seasonal_decompose(
            dataframe[column], model="Multiplicative"
        )
        mul_decomposition.plot()
        plt.show()


def stationarity_check(dataset: pd.DataFrame) -> float:
    '''
    Stationarity test by David Dickey and Wanye Fuller known as agumented fuller
    '''
    for column in dataset.columns:

        return sts.adfuller(dataset[column])


def check_correlation_factor(dataframe: pd.DataFrame, n_lags: int, type: str) -> None:
    '''
    input = dataframe
    n lags is the random int number
    for autocorrelation type = auto
    for partialcorrelation type = partial
    '''
    for column in dataframe.columns:
        if type == "auto":
            sgt.plot_acf(dataframe[column], lags = n_lags, zero = False)
        else:
            sgt.plot_pacf(dataframe[column], lags = n_lags, zero = False)
        plt.title(f"CF {column}")
        plt.show()
