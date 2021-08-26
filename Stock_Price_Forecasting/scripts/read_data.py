import pandas as pd

def read_dataset(path : str, file_type : str):
    '''
    input path as string, file_type = csv or xlsx
    -----------------------------------------------
    Read data from path with their respective file
    type and convet the data and time type and drop 
    SN 
    -----------------------------------------------
    Return the dataframe with index data and time
    '''
    if file_type == 'csv':
       dataframe = pd.read_csv(path)

    elif file_type == 'xlsx':
        dataframe = pd.read_excel(path)
    
    dataframe['Datetime'] = pd.to_datetime(dataframe['Date']).set_index('Datetime')
    dataframe.drop(['S.N.', 'Date'], axis = 1, inplace = True)

    return dataframe

def preprocess_data(dataset):
    '''
    '''
    dataset.where(dataset.values == 0, 
                other = (dataset.fillna(method = 'ffill') + 
                dataset.fillna(method = 'bfill'))/2)

    return dataset.sort_index()






