from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import pandas as pd 


def std_scale(X, X_test=None):
    """Standard scale dataframe and return as dataframe

    Args:
        X (pandas.DataFrame): Dataframe to be scaled

    Returns:
        pandas.DataFrame: Scaled dataframe
        StandardScaler: Scaling object for transforming if needed
    """    
    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X))
    if X_test is not None:
        X_test = pd.DataFrame(sc.transform(X_test), columns=X_test) 

    return X, X_test


def fill_metrics(met, df, X_cols):
    
    print(f'============\n{met}\n------------')
    
    train = df[~df[met].isnull()]
    predict = df[df[met].isnull()]
    
    y = train[met]
    X = train[X_cols]
    
    sc = StandardScaler()

    X_sc = sc.fit_transform(X)
    pred_sc = sc.transform(predict[X_cols])
    pred_sc = pd.DataFrame(pred_sc, columns=X_cols)

    lr = LinearRegression()
    lr.fit(X_sc, y)
    print('R2 Score', round(lr.score(X_sc, y), 3))
    print(pd.Series(lr.coef_, index=X.columns))
    
    df.loc[df[met].isnull(), met] = lr.predict(pred_sc)
    
    return df


def reverse_metrics(met, train, predict, X_cols):
    
    print(f'============\n{met}\n------------')
    
    y = train[met]
    X = train[X_cols]
    
    sc = StandardScaler()

    X_sc = sc.fit_transform(X)
    pred_sc = sc.transform(predict[X_cols])
    pred_sc = pd.DataFrame(pred_sc, columns=X_cols)

    lr = LinearRegression()
    lr.fit(X_sc, y)
    print('R2 Score', round(lr.score(X_sc, y), 3))
    print(pd.Series(lr.coef_, index=X.columns))
    
    predict[met] = lr.predict(pred_sc)
    
    return predict

