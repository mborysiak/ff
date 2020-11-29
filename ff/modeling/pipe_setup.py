from ff.modeling.data_setup import DataSetup

from sklearn.pipeline import Pipeline

# feature selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, SelectKBest,SelectPercentile, mutual_info_regression, f_regression
from sklearn.cluster import FeatureAgglomeration

# import all various models
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import LinearSVR, LinearSVC

class FeatureExtractionSwitcher(BaseEstimator):

    def __init__(self, estimator=FeatureAgglomeration()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.estimator = estimator


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def transform(self, X, y=None):
        return self.estimator.transform(X)


class FeatureDrop(BaseEstimator,TransformerMixin):

    def __init__(self, col=[]):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.col = col


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        if self.col is None:
            return X
        else:
            return X.drop(self.col, axis=1)


class FeatureSelect(BaseEstimator, TransformerMixin):

    def __init__(self, col=''):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 
        self.col = col


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        return X.loc[:, [self.col]]



class PipeSetup(DataSetup):

    def __init__(self, data, model_obj):
        self.data = data
        self.model_obj = model_obj

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
            'pca': PCA(),
            'feature_select': FeatureSelect(),
            'feature_drop': FeatureDrop(),
            'select_perc': SelectPercentile(score_func=f_regression, percentile=10),
            'select_from_model': SelectFromModel(estimator=Ridge()),
            'agglomeration': FeatureAgglomeration(),
            'k_best': SelectKBest(score_func=f_regression, k=10),
            'feature_switcher': FeatureExtractionSwitcher()
        }
        num_subset = {}
        for k in num_steps:
            num_subset[k] = num_options[k]
        
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
        """Function that returns model object based on model string name.

        Args:
            model ([type]): [description]

        Returns:
            [type]: [description]
        """        
        
        if self.model_obj == 'reg':

            model_options = {
                'lr': LinearRegression(),
                'ridge': Ridge(),
                'lasso': Lasso(),
                'enet': ElasticNet(),
                'rf': RandomForestRegressor(n_jobs=1),
                'xgb': XGBRegressor(n_jobs=1),
                'lgbm': LGBMRegressor(n_jobs=1),
                'knn': KNeighborsRegressor(n_jobs=1),
                'gbm': GradientBoostingRegressor(),
                'svr': LinearSVR()
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

    
    def stack_pipe(self, pipes, pipe_params):

        ests = []
        for i, p in enumerate(pipes):
            ests.append((f'p{i}', p))

        if self.model_obj == 'reg':
            stack = VotingRegressor(estimators=ests)
        elif self.model_obj == 'class':
            stack = VotingClassifier(estimators=ests)

        stack_params = {}
        for i, p in enumerate(pipe_params):
            for k, v in p.items():
                stack_params[f'stack__p{i}__{k}'] = v
        
        stack_pipe = Pipeline([('stack', stack)])

        return stack_pipe, stack_params



# from sklearn.base import BaseEstimator, TransformerMixin
# class CorrRemoval(BaseEstimator, TransformerMixin):

#     import pandas as pd

#     #Class Constructor
#     def __init__(self, corr_cutoff=0.02, collinear_cutoff=0.9):
#          self.corr_cutoff = corr_cutoff
#          self.collinear_cutoff = collinear_cutoff

#     @staticmethod
#     def filter_cut(corr, cut):
#         return list(abs(corr[abs(corr) > cut]).sort_values(ascending=False).index)[1:]
        
#     #Return self, nothing else to do here
#     def fit(self, X, y):

#         if isinstance(X, tuple):
#             X, y = X
#         super().fit(X=X, y=y)
   
    
#     #Custom transform method we wrote that creates aformentioned features and drops redundant ones 
#     def transform(self, X, y=None):

#         # calculate corr coefficients and put into dataframe matrix
#         corrs = np.corrcoef(X, y, rowvar=False)
#         corrs = pd.DataFrame(corrs, columns=range(corrs.shape[1]), index=range(corrs.shape[0]))
        
#         # extract out correlations with y variable
#         y_corr = corrs.iloc[:,-1]

#         # extract out the initial good columns based on correlation
#         good_cols = self.filter_cut(y_corr, self.corr_cutoff)

#         # create empty lists for best and bad cols
#         best_cols = []
#         bad_cols = []

#         # loop through each remaining column in order of correlation with the target
#         for col in good_cols:

#             # if the column is already listed in the bad columns, skip to the next
#             if col in bad_cols:
#                 continue

#             else:
#                 # if it's not listed in bad columns, add it to good columns
#                 best_cols.append(col)

#                 # remove any columns that are highly correlated with the current good column by adding to bad columns
#                 current_corr = corrs[col]
#                 bad_cols.extend(self.filter_cut(current_corr, self.collinear_cutoff))
    
#         self.best_cols = best_cols
        
#         X = X[:, self.best_cols]
#         print('post shape', X.shape)
#         return X