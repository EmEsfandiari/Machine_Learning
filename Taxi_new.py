import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
import numpy as np
from math import radians
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import metrics
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

taxi_train = pd.read_csv('taxi_train.csv')
taxi_test = pd.read_csv('taxi_test.csv')

# print(taxi_train.columns.to_list())

taxi_train['fare_amount'] = taxi_train['fare_amount'].mask(taxi_train['fare_amount'] <= 1, np.NaN)
taxi_train['pickup_longitude'] = taxi_train['pickup_longitude'].mask(taxi_train['pickup_longitude'] <= -75, np.NaN)
taxi_train['pickup_longitude'] = taxi_train['pickup_longitude'].mask(taxi_train['pickup_longitude'] >= -73, np.NaN)
taxi_train['dropoff_longitude'] = taxi_train['dropoff_longitude'].mask(taxi_train['dropoff_longitude'] <= -75, np.NaN)
taxi_train['dropoff_longitude'] = taxi_train['dropoff_longitude'].mask(taxi_train['dropoff_longitude'] >= -73, np.NaN)
taxi_train['pickup_latitude'] = taxi_train['pickup_latitude'].mask(taxi_train['pickup_latitude'] <= 40, np.NaN)
taxi_train['pickup_latitude'] = taxi_train['pickup_latitude'].mask(taxi_train['pickup_latitude'] >= 42, np.NaN)
taxi_train['dropoff_latitude'] = taxi_train['dropoff_latitude'].mask(taxi_train['dropoff_latitude'] <= 40, np.NaN)
taxi_train['dropoff_latitude'] = taxi_train['dropoff_latitude'].mask(taxi_train['dropoff_latitude'] >= 42, np.NaN)

taxi_train['pickup_datetime'] = pd.to_datetime(taxi_train['pickup_datetime'], utc=True)

taxi_train['hour'] = taxi_train['pickup_datetime'].dt.hour
taxi_train['day'] = taxi_train['pickup_datetime'].dt.dayofweek
taxi_train['month'] = taxi_train['pickup_datetime'].dt.month
taxi_train['year'] = taxi_train['pickup_datetime'].dt.year

taxi_train.dropna(axis=0, inplace=True)


def haversine(row):
    lon1 = row['pickup_longitude']
    lat1 = row['pickup_latitude']
    lon2 = row['dropoff_longitude']
    lat2 = row['dropoff_latitude']
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


taxi_train['distance'] = taxi_train.apply(lambda row: haversine(row), axis=1)

taxi_train['distance'] = taxi_train['distance'].mask(taxi_train['distance'] == 0, np.NaN)
taxi_train['distance'] = taxi_train['distance'].mask(taxi_train['distance'] > 60, np.NaN)
taxi_train['fare_amount'] = taxi_train['fare_amount'].mask(taxi_train['fare_amount'] > 100, np.NaN)

taxi_train.dropna(axis=0, inplace=True)

features = ['hour', 'day', 'month', 'year', 'distance', 'dropoff_latitude', 'pickup_latitude', 'pickup_longitude',
            'passenger_count']

X_train, X_val, y_train, y_val = train_test_split(taxi_train[features], taxi_train['fare_amount'], test_size=0.3)

params = {'xgb__learning_rate': [0.08, 0.1], 'xgb__n_estimators': [100, 500]}
steps = [('scaler', StandardScaler()),
         ('xgb', XGBRegressor(max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method='gpu_hist'))]
pipeline = Pipeline(steps)
cv = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='neg_mean_squared_error', refit=True, n_jobs=-1)
cv.fit(taxi_train[features].values, taxi_train['fare_amount'].values)
print(np.sqrt(-cv.best_score_))
print(cv.best_params_)
