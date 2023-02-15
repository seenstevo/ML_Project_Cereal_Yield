import pickle
import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_percentage_error


# set directory to this script

os.chdir(os.path.basename(os.path.dirname(__file__)))


# read in the data for models

data = pd.read_parquet('./data/data_for_model.pq')


# Split Train Test

X_train = data[data['Year'] < 2014].drop(columns = ['Country_Name', 'Country_Code', 'Cereal_Yield', 'Year'])
X_test = data[data['Year'] > 2013].drop(columns = ['Country_Name', 'Country_Code', 'Cereal_Yield', 'Year'])
y_train = data[data['Year'] < 2014]['Cereal_Yield']
y_test = data[data['Year'] > 2013]['Cereal_Yield']


# Load in the models

with open('./models/random_forest_regressor_model_cv.pickle', 'rb') as f:
    rfr_cv = pickle.load(f)
    
with open('./models/random_forest_regressor_model_full.pickle', 'rb') as f:
    rfr_full = pickle.load(f)
    
    
# Make the predictions

y_pred_test_cv = pd.Series(rfr_cv.predict(X_test), name = 'Predictions_Test_CV')
y_pred_train_cv = pd.Series(rfr_cv.predict(X_train), name = 'Predictions_Train_CV')

y_pred_test_full = pd.Series(rfr_full.predict(X_test), name = 'Predictions_Test_Full')
y_pred_train_full = pd.Series(rfr_full.predict(X_train), name = 'Predictions_Train_Full')

# Report the MAPE values

mape_test_cv = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred_test_cv)) * 100
mape_train_cv = mean_absolute_percentage_error(np.exp(y_train), np.exp(y_pred_train_cv)) * 100

mape_test_full = mean_absolute_percentage_error(np.exp(y_test), np.exp(y_pred_test_full)) * 100
mape_train_full = mean_absolute_percentage_error(np.exp(y_train), np.exp(y_pred_train_full)) * 100

print('MAPE Test CV Model:', mape_test_cv)
print('MAPE Train CV Model:', mape_train_cv)

print('MAPE Test Full Model:', mape_test_full)
print('MAPE Train Full Model:', mape_train_full)


# Combine into dataframe the predictions for Test

X_test_year_country = data[data['Year'] > 2013][['Year', 'Country_Name']].reset_index(drop = True)
y_test = y_test.reset_index(drop = True)

test_predictions_df = pd.concat([X_test_year_country, np.exp(y_test), np.exp(y_pred_test_cv), np.exp(y_pred_test_full)], axis = 1)
# Create an absolute difference percentage column
test_predictions_df['Absolute_Percent_Error'] = abs(test_predictions_df['Cereal_Yield'] - test_predictions_df['Predictions_Test_CV']) / test_predictions_df['Cereal_Yield'] * 100


# Combine into dataframe the predictions for Train

X_train_year_country = data[data['Year'] < 2014][['Year', 'Country_Name']].reset_index(drop = True)
y_train = y_train.reset_index(drop = True)

train_predictions_df = pd.concat([X_train_year_country, np.exp(y_train), np.exp(y_pred_train_cv), np.exp(y_pred_train_full)], axis = 1)
# Create an absolute difference percentage column
train_predictions_df['Absolute_Percent_Error'] = abs(train_predictions_df['Cereal_Yield'] - train_predictions_df['Predictions_Train_CV']) / train_predictions_df['Cereal_Yield'] * 100

# Report custom median absolute percentage error
print('Median Absolute Percentage Error Test:', test_predictions_df['Absolute_Percent_Error'].median())
print('Median Absolute Percentage Error Train:', train_predictions_df['Absolute_Percent_Error'].median())



# save the dataframe of predictions

test_predictions_df.to_csv('./data/test_predictions.csv', index = False)
train_predictions_df.to_csv('./data/train_predictions.csv', index = False)