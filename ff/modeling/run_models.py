from ff.modeling.data_setup import DataSetup
from ff.modeling.pipe_setup import PipeSetup
import pandas as pd
import numpy as np

# model and search setup
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import matthews_corrcoef, f1_score

class SciKitModel(PipeSetup):
    
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


    def default_params(self, model_name, num_steps={}):
        """Function that returns default search parameters for pipe components

        Args:
            model_name (str): Abbreviation of model in pipe.
                              Model Name Options: 
                                       'ridge' = Ridge() params,
                                       'lasso' = Lasso() params,
                                       'enet' = ElasticNet() params,
                                       'rf' = RandomForestRegressor() or Classifier() params,
                                       'lgbm' = LGBMRegressor() or Classifier() params,
                                       'xgb' = XGBRegressor() or Classifier() params,
                                       'knn' = KNeighborsRegress() or Classifier() params,
                                       'svr' = LinearSVR() params

                              Numeric Param Options:
                                       'agglomeration' = FeatureAgglomeration() params
                                       'pca' = PCA() params,
                                       'k_best' = SelectKBest params,
                                        

            num_steps (dict, optional): [description]. Defaults to {}.

        Returns:
            [type]: [description]
        """        

        num_params = {
            'agglomeration': {'n_clusters': Integer(2, 30)},
            'pca': {'n_components': Integer(1, 30)},
            'k_best': {'k': Integer(1, 50)},
            'select_perc': {'percentile': Real(0.02, 0.5)},
            'select_from_model': {'estimator': [Ridge(alpha=0.1), Ridge(alpha=1), Ridge(alpha=10),
                                                Lasso(alpha=0.1), Lasso(alpha=1), Lasso(alpha=10),
                                                RandomForestRegressor(max_depth=5), 
                                                RandomForestRegressor(max_depth=10)]},
            'feature_drop': {'col': Categorical(['avg_pick', None])}
        }

        model_params = {
            'ridge': {'alpha': Real(0.1, 1000, prior='log-uniform')},
            'lasso': {'alpha': Real(0.1, 1000, prior='log-uniform')},
            'enet': {'alpha': Real(0.1, 1000, prior='log-uniform'),
                     'l1_ratio': Real(0.5, 0.95)},
            'rf': {'n_estimators': Integer(50, 250),
                   'max_depth': Integer(2, 25),
                   'min_samples_leaf': Integer(1, 10),
                   'max_features': Real(0.1, 1)},
            'lgbm': {'n_estimators': Integer(25, 250),
                     'max_depth': Integer(2, 50),
                     'learning_rate': Real(10**-5, 1, 'log-uniform'),
                     'colsample_bytree': Real(0.05, 1),
                     'subsample': Real(0.05, 1),
                     'min_child_weight': Real(0.1, 100, 'log_uniform'),
                     'reg_lambda': Real(0.001, 1000, 'log_uniform'),
                     'reg_alpha': Real(0.001, 10, 'log_uniform')},
            'xgb': {'n_estimators': Integer(25, 250),
                     'max_depth': Integer(2, 50),
                     'learning_rate': Real(10**-5, 1, 'log-uniform'),
                     'colsample_bytree': Real(0.05, 1),
                     'subsample': Real(0.05, 1),
                     'min_child_weight': Real(0.1, 100, 'log_uniform'),
                     'reg_lambda': Real(0.001, 1000, 'log_uniform'),
                     'reg_alpha': Real(0.001, 10, 'log_uniform')},
            'knn': {'n_neighbors': Integer(2, 20),
                    'weights': Categorical(['distance', 'uniform']),
                    'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'])},
            'svr': {'C': Real(0.0001, 1000, 'log_uniform')}

        }

        if self.num_and_cat:
            num_prefix = 'preprocessor__numeric'
            cat_prefix = 'preprocessor__cat'
        else:
            num_prefix = 'preprocessor'
            cat_prefix = 'preprocessor'

        # initialize the parameter dictionary
        params = {}

        # add numeric processing params to dictionary
        for ns in num_steps:
            try:
                params_tmp = num_params[ns]
                for k, v in params_tmp.items():
                    params[f'{num_prefix}__{ns}__{k}'] = v
            except: pass

        # add model parameters to the dictionary 
        params_tmp = model_params[model_name]
        for k, v in params_tmp.items():
            params[f'model__{k}'] = v

        return params


    def scorer(self, score_type):

        scorers = {
            'r2': make_scorer(r2_score, greater_is_better=True),
            'mse': make_scorer(mean_squared_error, greater_is_better=False),
            'mae': make_scorer(mean_absolute_error, greater_is_better=False),
            'matt_coef': make_scorer(matthews_corrcoef, greater_is_better=True),
            'f1': make_scorer(f1_score, greater_is_better=True)
        }

        return scorers[score_type]

    def fit_opt(self, opt_model):
        
        opt_model.fit(self.X, self.y)
        return opt_model.best_estimator_

    def grid_search(self, pipe_to_fit, params, cv=5, scoring='neg_mean_squared_error'):

        search = GridSearchCV(pipe_to_fit, params, scoring=scoring, refit=True)
        best_model = self.fit_opt(search)

        return best_model


    def random_search(self, pipe_to_fit, params, cv=5, n_iter=50, scoring='neg_mean_squared_error'):

        search = RandomizedSearchCV(pipe_to_fit, params, n_iter=n_iter, scoring=scoring, refit=True)
        best_model = self.fit_opt(search)

        return best_model


    def bayes_search(self, pipe_to_fit, params, cv=5, n_iter=50, random_starts=25, 
                     scoring='neg_mean_squared_error', verbose=0):

        search = BayesSearchCV(pipe_to_fit, params, n_iter=n_iter, scoring=scoring, refit=True,
                                cv=cv, optimizer_kwargs={'n_initial_points': random_starts}, verbose=verbose)
        best_model = self.fit_opt(search)

        return best_model


    def cv_score(self, model, cv=5, scoring='neg_mean_squared_error', 
                 n_jobs=-1, return_mean=True):

        score = cross_val_score(model, self.X, self.y, cv=cv, n_jobs=-1, scoring=scoring)
        if return_mean:
            score = np.mean(score)
        return score


    def cv_predict(self, model, cv=5, n_jobs=-1):
        pred = cross_val_predict(self.pipe, self.X, self.y, cv=cv, n_jobs=n_jobs)
        return pred


    def cv_time_splits(self, col, min_idx):

        X_sort = self.X.sort_values(by=col).reset_index(drop=True)

        ts = X_sort[col].unique()

        cv_time = []
        for t in ts[min_idx:]:
            train_idx = list(X_sort[X_sort[col] < t].index)
            test_idx = list(X_sort[X_sort[col] == t].index)
            cv_time.append((train_idx, test_idx))

        return cv_time

    
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


    def val_scores(self, model, cv):

        if self.model_obj == 'reg':
            mse = self.cv_score(model, cv=5, scoring=self.scorer('mse'))
            r2 = self.cv_score(model, cv=5, scoring=self.scorer('r2'))
            
            for v, m in zip(['Val MSE:', 'Val R2:'], [mse, r2]):
                print(v, round(m, 3))

            return mse, r2

    def test_scores(self, model, X_test, y_test):

        if self.model_obj == 'reg':
            mse = mean_squared_error(y_test, model.predict(X_test))
            r2 = r2_score(y_test, model.predict(X_test))
            
            for v, m in zip(['Test MSE:', 'Test R2:'], [mse, r2]):
                print(v, round(m, 3))

            return mse, r2

    def return_labels(self, cols, time_or_all='time'):
        
        if time_or_all=='time':
            return self.data.loc[self.test_indices, cols]

        elif time_or_all=='all':
            return self.data.loc[:, cols]



