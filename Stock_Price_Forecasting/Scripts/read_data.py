import pandas as pd
import os

def read_dataset(path: str, file_type : str = 'csv') -> pd.DataFrame:

    """
    input path as string
    -----------------------------------------------
    Read data from path with their respective file
    type and convet the data and time type and drop
    SN
    -----------------------------------------------
    Return the dataframe with index data and time
    """

    if file_type == "csv":
        dataframe = pd.read_csv(path)

    elif file_type == "xlsx":
        dataframe = pd.read_excel(path)
        dataframe.drop(['S.N.','High','Low','Volume'], axis = 1, inplace = True)

    dataframe["Date"] = pd.to_datetime(dataframe["Date"])    
    dataframe = dataframe.set_index("Date")
    
    return dataframe

def join_dataframe(dataframe_1: pd.DataFrame, dataframe_2: pd.DataFrame) -> pd.DataFrame:
    """This join the two different dataframe"""
    dataframe =  dataframe_1.join(dataframe_2).sort_index()

    return(dataframe.groupby(dataframe.index.date).mean())


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """This function remove the nan values in dataset by using 
    mean of forward and backward values """
    dataframe.where(
        dataframe.values == 0,
        other = (dataframe.fillna(method = "ffill") + dataframe.fillna(method = "bfill")) / 2,
    )

    return dataframe.sort_index()
