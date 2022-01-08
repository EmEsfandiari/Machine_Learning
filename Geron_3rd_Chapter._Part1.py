import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.scatter(X, y)
plt.show()


# Normal Equation: theta_best = (Xtr * X) ** (-1) * Xtr *y
X_b = np.c_[np.ones((100, 1)), X]  # adds x0 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
print(X_new_b)

y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()


# Performing Linear Regression from scikit_learn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_)
y_predict_lin = lin_reg.predict(X_new)
print(y_predict_lin)


# Gradient Descent Step
eta = 0.1  # learning rate
n_iterations = 1000
m = 100
theta = np.random.randn(2, 1)  # random initialization

for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)  # Equation 4-5
    theta = theta - eta * gradients  # Equation 4-7

print(theta)


# Stochastic Gradient Descent
n_epochs = 50
t0, t1 = 5, 50  # learning schedule hyperparameters
m = 10  # the number of training instances


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients

print(theta)


# Linear Regression using Stochastic Gradient Descent
from sklearn.linear_model import SGDRegressor

sgd_r = SGDRegressor(eta0=0.1, penalty=None)
sgd_r.fit(X, y.ravel())

print(sgd_r.intercept_, sgd_r.coef_)
