import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
import numpy as np
from math import radians
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn import metrics

# print(metrics.SCORERS.keys())

taxi_train = pd.read_csv('taxi_train.csv')
taxi_test = pd.read_csv('taxi_test.csv')


# EDA
# print(taxi_train.info())
print(taxi_train.columns.to_list())
# print(taxi_test.columns.to_list())
# taxi_train.fare_amount.hist(bins=300, alpha=0.5)
# plt.xlim([-10, 800])
# plt.ylim([0,10])
# plt.show()
# print(taxi_train.shape)
# print(taxi_test.shape)
# print(taxi_train[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']].describe())
# print(taxi_train.head())
# print(taxi_test.head())
# print(taxi_train.passenger_count.value_counts())


# def haversine(row):
#     lon1 = row['pickup_longitude']
#     lat1 = row['pickup_latitude']
#     lon2 = row['dropoff_longitude']
#     lat2 = row['dropoff_latitude']
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arcsin(np.sqrt(a))
#     km = 6367 * c
#     return km
#
#
# # Feature Engineering
# taxi = pd.concat([taxi_train, taxi_test], )
# taxi['pickup_date'] = pd.to_datetime(taxi['pickup_datetime'])
# taxi['Day_of_Week'] = taxi['pickup_date'].dt.dayofweek
# taxi['hour'] = taxi['pickup_date'].dt.hour
# taxi['distance'] = taxi.apply(lambda row: haversine(row), axis=1)
# taxi_train = taxi[:20000]
# taxi_test = taxi[20000:].drop('fare_amount', axis=1)
# # print(taxi_train.isnull().sum())
# # print(taxi_test.isnull().sum())
# # print(taxi_train.shape)
# # print(taxi_test.shape)
#
# # # EDA
# # plt.scatter(x=taxi_train['fare_amount'], y=taxi_train['distance'], alpha=0.25)
# # plt.xlabel('fare amount')
# # plt.ylabel('distance')
# # plt.ylim(0,50)
# # plt.show()
# # hour_price = taxi_train.groupby('hour', as_index=False)['fare_amount'].median()
# # plt.plot(hour_price['hour'], hour_price['fare_amount'], marker= 'o')
# # plt.title('fare amount by hour of day')
# # plt.xlabel('hour of the day')
# # plt.ylabel('median of fare amount')
# # plt.show()
#
# # # Local Validation
# # skf = KFold(n_splits=5, shuffle=True, random_state=6666)
# # for train_index, test_index in skf.split(taxi_train):
# #     cv_train, cv_test = taxi_train.iloc[train_index], taxi_train.iloc[test_index]
# #     print(cv_train.shape)
# #     print(cv_test.shape)
#
# # Modeling
# features = ['hour', 'distance']
# X_train, X_val, y_train, y_val = train_test_split(taxi_train[features], taxi_train['fare_amount'], test_size=0.3)
# # lr_model = LinearRegression()
# # lr_model.fit(X_train, y_train)
# # predictions = lr_model.predict(X_val)
# # mse = mean_squared_error(y_val, predictions)
# # print(np.sqrt(mse))
#
# # sgd_model = SGDRegressor()
# # sgd_model.fit(X_train, y_train)
# # predictions = sgd_model.predict(X_val)
# # mse = mean_squared_error(y_val, predictions)
# # print(np.sqrt(mse))
#
# # gbr_model = GradientBoostingRegressor()
# # gbr_model.fit(X_train, y_train)
# # predictions = gbr_model.predict(X_val)
# # mse = mean_squared_error(y_val, predictions)
# # print(np.sqrt(mse))
#
# # ridge_model = Ridge()
# # ridge_model.fit(X_train, y_train)
# # predictions = ridge_model.predict(X_val)
# # mse = mean_squared_error(y_val, predictions)
# # print(np.sqrt(mse))
#
# rf_model = RandomForestRegressor()
# rf_model.fit(X_train, y_train)
# predictions = rf_model.predict(X_val)
# mse = mean_squared_error(y_val, predictions)
# print(np.sqrt(mse))
#
# # params = {'alpha': [0.01, 0.1, 1, 10]}
# # gs = GridSearchCV(estimator=Ridge(), param_grid=params, cv=5, scoring='neg_mean_squared_error', refit=True)
# # gs.fit(X_train, y_train)
# # print(np.sqrt(-gs.best_score_))
#
# # params = {'learning_rate': [0.01, 0.1, 10], 'max_depth': [3, 6, 9, 12]}
# # gs = GridSearchCV(estimator=GradientBoostingRegressor(), param_grid=params, cv=5, scoring='neg_mean_squared_error',
# #                   refit=True, n_jobs=-1)
# # gs.fit(taxi_train[features], taxi_train['fare_amount'])
# # print(gs.best_score_)
# # print(gs.best_params_)
#
# params = {'n_estimators': [10, 100, 1000]}
# gs = GridSearchCV(estimator=RandomForestRegressor(), param_grid=params, cv=5, scoring='neg_mean_squared_error',
#                   refit=True, n_jobs=-1)
# gs.fit(taxi_train[features], taxi_train['fare_amount'])
# print(np.sqrt(-gs.best_score_))
# print(gs.best_params_)
#
# # lr_model.fit(taxi_train[features], taxi_train['fare_amount'])
# # taxi_test['fare_amount'] = lr_model.predict(taxi_test[features])
# # taxi_submission = taxi_test[['id', 'fare_amount']]
# # print(taxi_submission.head())
