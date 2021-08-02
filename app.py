from flask import Flask, render_template , request
import joblib

dataset_url = 'https://github.com/JovianML/opendatasets/raw/master/data/house-prices-advanced-regression-techniques.zip'
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np

urlretrieve(dataset_url, 'house-prices.zip')
with ZipFile('house-prices.zip') as f:
    f.extractall(path='house-prices')
import os
data_dir = 'house-prices'

os.listdir(data_dir)

import pandas as pd
pd.options.display.max_columns = 200
pd.options.display.max_rows = 200

train_csv_path = data_dir + '/train.csv'
print(train_csv_path)

prices_df = pd.read_csv('house-prices/train.csv')
print(prices_df)

print(prices_df.info())

input_cols = prices_df.columns
input_cols =['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']
print(input_cols)


# Identify the name of the target column (a single string, not a list)
target_col = 'SalePrice'

inputs_df = prices_df[input_cols].copy()

targets = prices_df[target_col]

print(inputs_df)
print(targets)

numeric_cols = inputs_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(numeric_cols)

categorical_cols = inputs_df.select_dtypes('object').columns.tolist()
print(categorical_cols)

missing_counts = inputs_df[numeric_cols].isna().sum().sort_values(ascending=False)
print(missing_counts[missing_counts > 0])


from sklearn.impute import SimpleImputer


# 1. Create the imputer
imputer = SimpleImputer(strategy = 'mean')
# 2. Fit the imputer to the numeric colums
imputer.fit(inputs_df[numeric_cols])
# 3. Transform and replace the numeric columns
inputs_df[numeric_cols] =imputer.transform(inputs_df[numeric_cols] ) 

missing_counts = inputs_df[numeric_cols].isna().sum().sort_values(ascending=False)
print(missing_counts[missing_counts > 0]) # should be an empty list

print(inputs_df[numeric_cols].describe().loc[['min', 'max']])

from sklearn.preprocessing import MinMaxScaler
# Create the scaler
scaler = MinMaxScaler()

# Fit the scaler to the numeric columns
scaler.fit(inputs_df[numeric_cols])

# Transform and replace the numeric columns
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])

print(inputs_df[numeric_cols].describe().loc[['min', 'max']])

print(inputs_df[categorical_cols].nunique().sort_values(ascending=False))

from sklearn.preprocessing import OneHotEncoder

# 1. Create the encoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# 2. Fit the encoder to the categorical colums
encoder.fit(inputs_df[categorical_cols])
print(encoder.categories_)

# 3. Generate column names for each category
encoded_cols = list(encoder.get_feature_names(categorical_cols))
len(encoded_cols)

# 4. Transform and add new one-hot category columns
inputs_df[encoded_cols] = encoder.transform(inputs_df[categorical_cols])

print(inputs_df)

from sklearn.model_selection import train_test_split

train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs_df[numeric_cols + encoded_cols], 
                                                                        targets, 
                                                                        test_size=0.25, 
                                                                        random_state=42)

print(train_inputs)
print(train_targets)
print('val inputs' , val_inputs)
print('val_targets' ,val_targets)

from sklearn.linear_model import Ridge
# Create the model
model = Ridge(solver='auto')
# Fit the model using inputs and targets
model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)

from sklearn.metrics import mean_squared_error
train_preds = model.predict(train_inputs)
print(train_preds)

import math
train_rmse = math.sqrt(mean_squared_error(train_targets, train_preds))
print(train_rmse)

import math
train_rmse = math.sqrt(mean_squared_error(train_targets, train_preds))

print('The RMSE loss for the training set is $ {}.'.format(train_rmse))

val_preds = model.predict(val_inputs)

print(val_preds)

val_rmse = math.sqrt(mean_squared_error(val_targets, val_preds))
print('The RMSE loss for the validation set is $ {}.'.format(val_rmse))

weights = model.coef_
print(weights)

weights_df = pd.DataFrame({
    'columns': train_inputs.columns,
    'weight': weights
}).sort_values('weight', ascending=False)

print(weights_df)



def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols].values)
    X_input = input_df[numeric_cols + encoded_cols]
    return model.predict(X_input)[0]

sample_input = { 'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 77.0, 'LotArea': 9320,
 'Street': 'Pave', 'Alley': None, 'LotShape': 'IR1', 'LandContour': 'Lvl', 'Utilities': 'AllPub',
 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'NAmes', 'Condition1': 'Norm', 'Condition2': 'Norm',
 'BldgType': '1Fam', 'HouseStyle': '1Story', 'OverallQual': 4, 'OverallCond': 5, 'YearBuilt': 1959,
 'YearRemodAdd': 1959, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'Plywood',
 'Exterior2nd': 'Plywood', 'MasVnrType': 'None','MasVnrArea': 0.0,'ExterQual': 'TA','ExterCond': 'TA',
 'Foundation': 'CBlock','BsmtQual': 'TA','BsmtCond': 'TA','BsmtExposure': 'No','BsmtFinType1': 'ALQ',
 'BsmtFinSF1': 569,'BsmtFinType2': 'Unf','BsmtFinSF2': 0,'BsmtUnfSF': 381,
 'TotalBsmtSF': 950,'Heating': 'GasA','HeatingQC': 'Fa','CentralAir': 'Y','Electrical': 'SBrkr', '1stFlrSF': 1225,
 '2ndFlrSF': 0, 'LowQualFinSF': 0, 'GrLivArea': 1225, 'BsmtFullBath': 1, 'BsmtHalfBath': 0, 'FullBath': 1,
 'HalfBath': 1, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1,'KitchenQual': 'TA','TotRmsAbvGrd': 6,'Functional': 'Typ',
 'Fireplaces': 0,'FireplaceQu': np.nan,'GarageType': np.nan,'GarageYrBlt': np.nan,'GarageFinish': np.nan,'GarageCars': 0,
 'GarageArea': 0,'GarageQual': np.nan,'GarageCond': np.nan,'PavedDrive': 'Y', 'WoodDeckSF': 352, 'OpenPorchSF': 0,
 'EnclosedPorch': 0,'3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': np.nan, 'Fence': np.nan, 'MiscFeature': 'Shed',
 'MiscVal': 400, 'MoSold': 1, 'YrSold': 2010, 'SaleType': 'WD', 'SaleCondition': 'Normal'}

predicted_price = predict_input(sample_input)

print('The predicted sale price of the house is ${}'.format(predicted_price))

house_price_predictor = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}

joblib.dump(house_price_predictor, 'house_price_predictor.joblib')












# initialse the app
#app = Flask(__name__)


# if __name__ == '__main__':
#     app.run(debug = True)

