#%%

from ff.db_operations import DataManage
from ff import general
from ff.modeling.run_models import SciKitModel
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skopt.space import Real, Categorical, Integer

root_path = general.get_main_path('Fantasy_Football')
db_path = f'{root_path}/Data/Databases/'
dm = DataManage(db_path)

df = dm.read('''SELECT * FROM RB_Stats''', 'Season_Stats')
df = df[['player', 'rush_yds', 'year', 'team',  'age', 'avg_pick']]

skm = SciKitModel(df)
skm.Xy_split(y_metric='avg_pick', to_drop=['player'])

p = skm.model_pipe(model='ridge', 
                  num_steps=['std_scale', 'pca'], 
                  cat_steps=['one_hot']
                )

model_params = {'alpha': [1, 2, 3, 4, 5]}
num_params = {'pca__n_components': [1, 2, 3]}

best_score, g = skm.grid_search(model_params, num_params)
print(best_score)
best_score, r = skm.random_search(model_params, num_params, n_iter=3)
print(best_score)

# model_params = {'alpha': Integer(1, 5)}
# num_params = {'pca__n_components': Integer(1,2)}
# best_score, bayes = sm.bayes_search(model_params, num_params, n_iter=3)
# print(best_score)

X, y = skm.return_X_y()
cv_time = skm.cv_time_splits('year', min_idx=10)

best_score, best_model = skm.grid_search(model_params, num_params, cv=cv_time)
predictions = skm.cv_predict_time(best_model, cv_time)



# %%
