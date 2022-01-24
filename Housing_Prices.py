import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

housing_train = pd.read_csv('house_prices_train.csv')
housing_test = pd.read_csv('house_prices_test.csv')


# EDA
print(housing_train.head())
print(housing_train.shape)
print(housing_test.shape)
print(housing_train.columns.to_list())
print(housing_train.describe())
print(housing_train.isnull().sum())


def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values


def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index, dtype='float')
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature
    return train_feature.values


def mean_target_encoding(train, test, target, categorical, alpha=5):
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    # Return new features to add to the model
    return train_feature, test_feature


# Feature Engineering
housing = pd.concat([housing_train, housing_test])
housing['TotalArea'] = housing['TotalBsmtSF'] + housing['1stFlrSF'] + housing['2ndFlrSF']
housing['GardenArea'] = housing['LotArea'] - housing['1stFlrSF']
housing['TotalBath'] = housing['FullBath'] + housing['HalfBath']
le = LabelEncoder()
housing['CentralAir_enc'] = le.fit_transform(housing['CentralAir'])
# ohe = pd.get_dummies(housing['RoofStyle'], prefix='RoofStyle')
# housing = pd.concat([housing, ohe], axis=1)

housing_train = housing[housing.Id.isin(housing_train.Id)]
housing_test = housing[housing.Id.isin(housing_test.Id)]
housing_train['RoofStyle_enc'], housing_test['RoofStyle_enc'] = mean_target_encoding(train=housing_train,
                                                                                     test=housing_test,
                                                                                     target='SalePrice',
                                                                                     categorical='RoofStyle', alpha=10)
print(housing_test[['RoofStyle', 'RoofStyle_enc']].drop_duplicates())

# Local Validation
kfv = KFold(n_splits=5, shuffle=True, random_state=6666)
metrics = []
for train_index, test_index in kfv.split(housing_train):
    cv_train, cv_test = housing_train.iloc[train_index], housing_train.iloc[test_index]
    # Modeling
    rfr_model = RandomForestRegressor()
    rfr_model.fit(cv_train[
                      ['TotalBath', 'GardenArea', 'TotalArea', 'RoofStyle_enc', 'CentralAir_enc']],
                  cv_train['SalePrice'])
    predictions = rfr_model.predict(
        cv_test[['TotalBath', 'GardenArea', 'TotalArea', 'RoofStyle_enc', 'CentralAir_enc']])
    metric = np.sqrt(mean_squared_error(cv_test['SalePrice'], predictions))
    metrics.append(metric)
print(np.mean(metrics))
