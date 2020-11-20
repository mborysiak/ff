from ff.modeling.prepare import DataPrepare
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class SciKitModel(DataPrepare):
    
    def __init__(self, data, model_obj='reg', set_seed=1234):
        """Class the automates model sklearn pipeline creation and model training

        Args:
            data (pandas.DataFrame): Data to be modeled with features and target
            model_obj (str, optional): Model objective for regression (reg) or 
                                       classification (class). Defaults to 'reg'.
            set_seed (int, optional): Numpy random seed to set state. Defaults to 1234.
        """        
        self.data = data
        self.model_obj = model_obj
        np.random.seed(set_seed)

    @staticmethod
    def create_pipe(pipe_steps):
        """Internal function that accepts a dictionary and creates Pipeline

        Args:
            pipe_steps (dict, optional): Dictionary with key-value for steps in Pipeline.

        Returns:
            sklearn.Pipeline: Pipeline object with specified steps embedded.
        """
        step_list = []
        for k, v in pipe_steps.items():
            step_list.append((k, v))

        return Pipeline(steps=step_list)

    
    def _cat_pipe(self, cat_steps):
        """Helper function to create categorical transform pipeline.

        Args:
            cat_steps (list): List that contains desired steps for transformation.
                              Options: 'cat_impute' = SimpleImputer()
                                       'one_hot' = OneHotEncoder() 

        Returns:
            sklearn.Pipeline: Pipeline with categorical column transformations
        """        
        self.cat_features = self.X.select_dtypes(include=['object']).columns

        cat_options = {
            'cat_impute': SimpleImputer(),
            'one_hot': OneHotEncoder()
        }

        cat_subset = {k: v for k, v in cat_options.items() if k in cat_steps}
        
        return self.create_pipe(cat_subset)


    def _num_pipe(self, num_steps):
        """Helper function to create numerical transform pipeline.

        Args:
            num_steps (list): List that contains desired steps for transformation.
                              Options: 'num_impute' = SimpleImputer()
                                       'std_scale' = OneHotEncoder() 
                                       'min_max_scale' = MinMaxScaler()
                                       'pca' = PCA()

        Returns:
            sklearn.Pipeline: Pipeline with numeric column transformations
        """        
        self.num_features = self.X.select_dtypes(exclude=['object']).columns

        num_options = {
            'num_impute': SimpleImputer(),
            'std_scale': StandardScaler(),
            'min_max_scale': MinMaxScaler(),
            'pca': PCA()
        }

        num_subset = {k: v for k, v in num_options.items() if k in num_steps}
        
        return self.create_pipe(num_subset)

    
    def _column_transform(self, num_pipe, cat_pipe):
        """Function that combines numeric and categorical pipeline transformers

        Args:
            num_pipe (sklearn.Pipeline): Numeric transformation pipeline
            cat_pipe (sklearn.Pipeline): Categorical transformation pipeline

        Returns:
            sklear.ColumnTransformer: ColumnTransformer object with both numeric 
                                      and categorical trasnformation steps
        """        
        return ColumnTransformer(
                    transformers=[
                        ('numeric', num_pipe, self.num_features),
                        ('cat', cat_pipe, self.cat_features)
                    ])


    def get_model(self, model):
        
        if self.model_obj == 'reg':

            model_options = {
                'lr': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(),
                'rf': RandomForestRegressor()
            }

        elif self.model_obj == 'class':
             
             model_options = {
                 'logistic': LogisticRegression()
             }

        return model_options[model]


    def model_pipe(self, model, cat_steps=None, num_steps=None):
        
        self.num_and_cat = False
        if num_steps is not None and cat_steps is not None:
            num_pipe = self._num_pipe(num_steps)
            cat_pipe = self._cat_pipe(cat_steps)
            column_transform = self._column_transform(num_pipe, cat_pipe)
            self.num_and_cat = True

        elif num_steps is not None:
            column_transform = self._num_pipe(num_steps)

        elif cat_steps is not None:
            column_transform = self._cat_pipe(cat_steps)

        else:
            print('No feature processing specified')
            column_transform = None

        model = self.get_model(model)    
    
        if column_transform is not None:
            self.pipe = self.create_pipe({'preprocessor': column_transform,
                                          'model': model})
        else:
            self.pipe = self.create_pipe({'model': model})

        return self.pipe


    def cv_time_splits(self, col, min_idx):

        X_sort = self.X.sort_values(by=col).reset_index(drop=True)

        ts = X_sort[col].unique()

        cv_time = []
        for t in ts[min_idx:]:
            train_idx = list(X_sort[X_sort[col] < t].index)
            test_idx = list(X_sort[X_sort[col] == t].index)
            cv_time.append((train_idx, test_idx))

        return cv_time

    
    def _set_params(self, model_params={}, num_params={}, cat_params={}):
        
        model_prefix = 'model__'

        if self.num_and_cat:
            num_prefix = 'preprocessor__numeric__'
            cat_prefix = 'preprocessor__cat__'
        else:
            num_prefix = 'preprocessor__'
            cat_prefix = 'preprocessor__'

        params = {}

        for k, v in model_params.items():
            params[model_prefix + k] = v

        for k, v in num_params.items():
            params[num_prefix + k] = v
        
        for k, v in cat_params.items():
            params[cat_prefix + k] = v

        return params


    def fit_opt(self, opt_model):
        
        opt_model.fit(self.X, self.y)
        best_score = opt_model.best_score_
        best_model = opt_model.best_estimator_

        return best_score, best_model

    def grid_search(self, model_params={}, num_params={}, cat_params={}, 
                    cv=5, scoring='neg_mean_squared_error'):

        params = self._set_params(model_params, num_params, cat_params)
        search = GridSearchCV(self.pipe, params, scoring=scoring, refit=True)
        best_score, best_model = self.fit_opt(search)

        return best_score, best_model


    def random_search(self, model_params={}, num_params={}, cat_params={}, 
                      cv=5, n_iter=50, scoring='neg_mean_squared_error'):

        params = self._set_params(model_params, num_params, cat_params)              
        search = RandomizedSearchCV(self.pipe, params, n_iter=n_iter, scoring=scoring, refit=True)
        best_score, best_model = self.fit_opt(search)

        return best_score, best_model


    def bayes_search(self, model_params={}, num_params={}, cat_params={}, 
                     cv=5, n_iter=50, scoring='neg_mean_squared_error'):

        params = self._set_params(model_params, num_params, cat_params)
        search = BayesSearchCV(self.pipe, params, n_iter=n_iter, scoring=scoring, refit=True)
        best_score, best_model = self.fit_opt(search)

        return best_score, best_model


    def cv_score(self, model, cv=5, scoring='neg_mean_squared_error', 
                 n_jobs=-1, return_mean=True):
        score = cross_val_score(model, self.X, self.y, cv=cv, n_jobs=-1, scoring=scoring)
        if return_mean:
            score = np.mean(score)
        return score


    def cv_predict(self, model, cv=5, n_jobs=-1):
        pred = cross_val_predict(self.pipe, self.X, self.y, cv=cv, n_jobs=n_jobs)
        return pred

    
    def cv_predict_time(self, model, cv_time):

        predictions = []
        self.test_indices = []
        for tr, te in cv_time:
            
            X_train, y_train = self.X.iloc[tr, :], self.y[tr]
            X_test, _ = self.X.iloc[te, :], self.y[te]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            predictions.extend(pred)
            self.test_indices.extend(te)

        return predictions

    
    def return_labels(self, cols, time_or_all='time'):
        
        if time_or_all=='time':
            return self.data.loc[self.test_indices, cols]

        elif time_or_all=='all':
            return self.data.loc[:, cols]