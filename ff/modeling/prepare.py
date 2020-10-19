from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd 
import numpy as np


def Xy_split(df, y_metric):
    """Split train dataset into X and y by excluding non-numeric values

    Args:
        df (pandas.DataFrame): DataFrame containing samples features and target variable
        y_metric (str): Column name of target variable to split into y

    Returns:
        X (pandas.DataFrame): Features matrix with numeric columns
        y (pandas.DataFrame): Target vector containing corresponding measure
    """ 
    # select only numeric columns and drop the specific y-metric
    X = df.select_dtypes(exclude=['object'])

    # create the y variable based on input metric, if it exists
    if y_metric in df.columns: 
        X = X.drop(y_metric, axis=1)
        y = df[y_metric]
    else: 
        y = None
        print(f'{y_metric} not in dataframe to create y')

    return X, y


def Xy_split_list(df, y_metric, col_list):
    """Split train dataset into X and y by passing list of desired columns

    Args:
        df (pandas.DataFrame): DataFrame containing samples features and target variable
        y_metric (str): Column name of target variable to split into y
        col_list (str): List of column names to keep within X dataframe

    Returns:
        X (pandas.DataFrame): Features matrix with selected columns
        y (pandas.DataFrame): Target vector containing corresponding measure
    """ 
    # create the X variable based on column list
    X = df[col_list]
    
    # create the y variable based on input metric, if it exists
    if y_metric in df.columns: 
        y = df[y_metric]
    else: 
        y = None
        print(f'{y_metric} not in dataframe to create y')

    return X, y


def data_scale(X, X_test=None, sc=StandardScaler()):
    """Standard (or other) scale dataframe and return as dataframe

    Args:
        X (pandas.DataFrame): Dataframe to be scaled
        X_test (pandas.DataFrame, optional): Test dataframe to transform. Defaults to None.
        sc (preprocessing.Scale, optional): Scaling transform object. Defaults to StandardScaler().

    Returns:
        pandas.DataFrame: Scaled dataframe
        StandardScaler: Scaling object for transforming if needed
    """        
    X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
    if X_test is not None:
        X_test = pd.DataFrame(sc.transform(X_test), columns=X_test.columns) 

    return X, X_test


def pca_scale(X_train, X_val=None, n_comp=10, rseed=1234):
    """Perform PCA scaling on X_train dataset and validation if present

    Args:
        X_train (pandas.DataFrame): Training matrix dataset
        X_val (pandas.DataFrame, optional): Validation dataset. Defaults to None.
        n_comp (int, optional): [description]. Defaults to 10.

    Returns:
        X_train (pandas.DataFrame): PCA-transformed training dataset
        X_val (pandas.DataFrame): PCA-transformed validation dataset
    """    
    # ensure that n_components is less than number of columns and rows
    n_comp = np.min([X_train.shape[1], X_train.shape[0], n_comp])

    # fit the PCA object
    pca = PCA(n_components=n_comp, random_state=rseed, svd_solver='randomized')
    pca.fit(X_train)

    # transform train (and validation if present) and pass array back into dataframe
    X_train = pd.DataFrame(pca.transform(X_train), 
                           columns=['pca' + str(i) for i in range(n_comp)])
    if X_val is not None:
        X_val = pd.DataFrame(pca.transform(X_val), 
                             columns=['pca' + str(i) for i in range(n_comp)])

    return X_train, X_val


def one_hot(df, col, drop_first=True):
    """Add one-hot-encoded columns to pandas dataframe.

    Args:
        df (pandas.DataFrame): DataFrame with column to be encoded
        col (str): Column to be one-hot-encoded

    Returns:
        pandas.DataFrame: DataFrame with OHE column (and original dropped)
    """    
    if col in df.columns:
        return pd.concat([df, pd.get_dummies(df[col], drop_first=drop_first)], axis=1).drop(col, axis=1)
    else:
        print(f'{col} does not exist in dataframe') 


def count_null(df):
    """Count null values for each column in DataFrame

    Args:
        df (pandas.DataFrame): DataFrame to count nulls in each column
    """    
    print('Null Counts:', df.isnull().sum()[df.isnull().sum() > 0])