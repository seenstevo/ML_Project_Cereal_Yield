# Import genereal libraries
import pandas as pd
import numpy as np
import os
import pickle

# Imports from sklearn
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone


# set directory to this script

os.chdir(os.path.basename(os.path.dirname(__file__)))


# read in the data for models

data = pd.read_parquet('./data/data_for_model.pq')


# Split Train Test

X_train = data[data['Year'] < 2014].drop(columns = ['Country_Name', 'Country_Code', 'Cereal_Yield', 'Year'])
X_test = data[data['Year'] > 2013].drop(columns = ['Country_Name', 'Country_Code', 'Cereal_Yield', 'Year'])
y_train = data[data['Year'] < 2014]['Cereal_Yield']
y_test = data[data['Year'] > 2013]['Cereal_Yield']


# Train the model

pipe = Pipeline([('knn_imputer', KNNImputer()),
                 ('scaler', StandardScaler()), 
                 ('selectk', SelectKBest()), 
                 ('rf_reg', RandomForestRegressor())])

pipe_param = {
    "selectk__k" : np.arange(40, 50, 3),
    'rf_reg__max_depth': [10, 12],
    'rf_reg__min_samples_leaf': [2, 3]
}

cv = TimeSeriesSplit(n_splits=3)

gs_pipe = GridSearchCV(pipe,
                        pipe_param,
                        cv = cv,
                        scoring = 'neg_mean_absolute_percentage_error',
                        verbose = 2,
                        n_jobs = -1,
                        error_score = 0.0)

gs_pipe.fit(X_train, y_train)


# save the best model from cross validation and make a clone to retrain on the full train set

best_estimator_cv = gs_pipe.best_estimator_

best_estimator_full = clone(gs_pipe.best_estimator_)
best_estimator_full.fit(X_train, y_train)


# save the model to file for use in predictions


# model from cross validation 
filename = './models/random_forest_regressor_model_cv.pickle'

with open(filename, 'wb') as f:
    pickle.dump(best_estimator_cv, f)

# model re-trained on all training
filename = './models/random_forest_regressor_model_full.pickle'

with open(filename, 'wb') as f:
    pickle.dump(best_estimator_full, f)