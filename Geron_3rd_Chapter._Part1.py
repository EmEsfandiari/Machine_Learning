import numpy as np

# from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
#
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
#
# plt.scatter(X, y)
# plt.show()
#
# # Normal Equation: theta_best = (Xtr * X) ** (-1) * Xtr *y
# X_b = np.c_[np.ones((100, 1)), X]  # adds x0 to each instance
# theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)
#
# X_new = np.array([[0], [2]])
# X_new_b = np.c_[np.ones((2, 1)), X_new]
# print(X_new_b)
#
# y_predict = X_new_b.dot(theta_best)
# print(y_predict)
#
# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()
#
# # Performing Linear Regression from scikit_learn
# lin_reg = LinearRegression()
# lin_reg.fit(X, y)
# print(lin_reg.intercept_)
# y_predict_lin = lin_reg.predict(X_new)
# print(y_predict_lin)
#
# # Gradient Descent Step
# eta = 0.1  # learning rate
# n_iterations = 1000
# m = 100
# theta = np.random.randn(2, 1)  # random initialization
#
# for iteration in range(n_iterations):
#     gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # Equation 4-5
#     theta = theta - eta * gradients  # Equation 4-7
#
# print(theta)
#
# # Stochastic Gradient Descent
# n_epochs = 50
# t0, t1 = 5, 50  # learning schedule hyperparameters
# m = 10  # the number of training instances
#
#
# def learning_schedule(t):
#     return t0 / (t + t1)
#
#
# theta = np.random.randn(2, 1)
#
# for epoch in range(n_epochs):
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index:random_index + 1]
#         yi = y[random_index:random_index + 1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
#         eta = learning_schedule(epoch * m + i)
#         theta = theta - eta * gradients
#
# print(theta)
#
# # Linear Regression using Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor

#
# sgd_r = SGDRegressor(eta0=0.1, penalty=None)
# sgd_r.fit(X, y.ravel())
#
# print(sgd_r.intercept_, sgd_r.coef_)
#
# # Polinomial Regression
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.rand(m, 1)
#
# plt.scatter(X, y)
# plt.show()
#
from sklearn.preprocessing import PolynomialFeatures
#
# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly_features.fit_transform(X)
#
# print(X[0])
# print(X_poly[0])  # Square of X
#
# lin_reg = LinearRegression()
# lin_reg.fit(X_poly, y)
# print(lin_reg.intercept_, lin_reg.coef_)
#
from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
#
#
# def plot_learning_curve(model, X, y):
#     X_train, X_val, y_train, y_val = train_test_split(X, y)
#     train_errors, val_errors = [], []
#     for m in range(1, len(X_train)):
#         model.fit(X_train[:m], y_train[:m])
#         y_train_pred = model.predict(X_train[:m])
#         y_val_pred = model.predict(X_val)
#         train_errors.append(mean_squared_error(y_train[:m], y_train_pred))
#         val_errors.append(mean_squared_error(y_val, y_val_pred))
#     plt.plot(np.sqrt(train_errors), 'r-+', linewidth=2, label='train')
#     plt.plot(np.sqrt(val_errors), 'b-', linewidth=3, label='val')
#     plt.legend()
#     plt.ylim(0, 1)
#     return plt.show()
#
#
# lin_reg = LinearRegression()
# plot_learning_curve(lin_reg, X, y)
#
from sklearn.pipeline import Pipeline
#
# polynomial_regression = Pipeline([
#     ("ploy_features", PolynomialFeatures(degree=2, include_bias=False)),
#     ("lin_reg", LinearRegression())
# ])
# plot_learning_curve(polynomial_regression, X, y)
#
# # Regularized Linear Models
#
# # Ridge Regression
#
# from sklearn.linear_model import Ridge
#
# ridge_reg = Ridge(alpha=1, solver="cholesky") # Equation 4-10
# ridge_reg.fit(X, y)
# predict = ridge_reg.predict([[1.5]])
# print(predict)
#
# # Using Stochastic Gradient
# sgd_reg = SGDRegressor(penalty="l2")
# sgd_reg.fit(X, y.ravel())
# predict_2 = sgd_reg.predict([[1.5]])
# print(predict_2)


# Lasso Regression

from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)  # == SGDRegressor(penalty = "l1")  # Equation 4-11
lasso_reg.fit(X, y)
predict_3 = lasso_reg.predict([[1.5]])
print(predict_3)

from sklearn.linear_model import ElasticNet

elastic_reg = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_reg.fit(X, y)
predict_4 = elastic_reg.predict([[1.5]])
print(predict_4)

