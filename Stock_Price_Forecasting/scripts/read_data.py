import pandas as pd
import os


def read_dataset(path: str) -> pd.DataFrame:

    """
    input path as string
    -----------------------------------------------
    Read data from path with their respective file
    type and convet the data and time type and drop
    SN
    -----------------------------------------------
    Return the dataframe with index data and time
    """
    name, extension = os.path.splitext(path)
    if extension == ".csv":
        dataframe = pd.read_csv(path)

    elif extension == ".xlsx":
        dataframe = pd.read_excel(path)

    dataframe["Datetime"] = pd.to_datetime(dataframe["Date"]).set_index("Datetime")
    dataframe.drop(["Date"], axis = 1, inplace = True)

    return dataframe


def join_dataframe(dataframe_1: pd.DataFrame, dataframe_2: pd.DataFrame) -> pd.DataFrame:
    """ """
    return dataframe_1.join(dataframe_2).sort_index()


def preprocess_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ """
    dataframe.where(
        dataframe.values == 0,
        other = (dataframe.fillna(method="ffill") + dataframe.fillna(method="bfill")) / 2,
    )

    return dataframe.sort_index()
