import numpy as np

class MyLinearRegression():
    def __init__(self, theta):
        self.theta = theta

    def add_intercept(self, x):
        return np.c_[np.ones(x.shape[0]), x]

    def predict(self, x):
        return self.add_intercept(x) @ self.theta

    def cost_elem_(self, x, y):
        dif = self.predict(x) - y.reshape(x.shape)
        return dif ** 2 / (2 * y.shape[0])

    def cost_(self, x, y):
        dif = self.predict(x) - y.reshape(x.shape)
        return dif @ dif / (2 * y.shape[0])

    def fit(self, x, y, alpha, n_cycle):
        y = y.flatten()
        pre_cal = self.add_intercept(x)
        for _ in range(n_cycle):
            nabla = (pre_cal.T @ ((pre_cal @ self.theta)- y) ) / (y.shape[0])
            # self.theta -= alpha * nabla
            self.theta[0] -= alpha * nabla[0]
            self.theta[1] -= alpha * nabla[1]
