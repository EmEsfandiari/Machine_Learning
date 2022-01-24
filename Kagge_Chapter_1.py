import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

train = pd.read_csv('demand_forecasting_train_1_month.csv')
test = pd.read_csv('demand_forecasting_test.csv')

# EDA
print('Train:', train.columns.tolist())
print('Test:', test.columns.tolist())
print(train.head())
train.sales.hist(bins=30, alpha=0.5)
plt.show()

# Local Validation
tss = TimeSeriesSplit(5)
train = train.sort_values('date')
metrics = []
for train_index, test_index in tss.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    # Modeling
    rfr_model = RandomForestRegressor()
    rfr_model.fit(cv_train[['store', 'item']], cv_train['sales'])
    predictions = rfr_model.predict(cv_test[['store', 'item']])
    metric = mean_squared_error(cv_test['sales'], predictions)
    metrics.append(metric)
print(metrics)
metrics_mean = np.mean(metrics)
metrics_std = np.std(metrics)
metrics_min, metrics_max = metrics_mean - metrics_std, metrics_mean + metrics_std
print(metrics_min, metrics_max)

# rfr_model = RandomForestRegressor()
# rfr_model.fit(train[['store', 'item']], train['sales'])
# test['sales'] = rfr_model.predict(test[['store', 'item']])
# test[['id', 'sales']].to_csv('kaggle_submission.csv', index=False)
