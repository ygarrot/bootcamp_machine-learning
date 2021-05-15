import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyLR():
    def __init__(self, theta, alpha=5e-5, n_cycle=320000):
        self.theta = theta
        self.alpha = alpha
        self.n_cycle = n_cycle

    def add_intercept(self, x):
        return np.c_[np.ones(x.shape[0]), x]

    def predict_(self, x):
        return self.add_intercept(x) @ self.theta

    def cost_elem_(self, x, y):
        dif = self.predict_(x) - y
        return dif ** 2 / (2 * y.shape[0])

    def mse_(self, x, y):
        dif = (self.predict_(x) - y).flatten()
        return dif @ dif / y.shape[0]

    def cost_(self, x, y):
        dif = (self.predict_(x) - y).flatten()
        return dif @ dif / (2 * y.shape[0])

    def fit_(self, x, y):
        pre_cal = self.add_intercept(x)
        y = y.flatten()
        for _ in range(self.n_cycle):
            nabla = (pre_cal.T @ ((pre_cal @ self.theta) - y)) / (y.shape[0])
            self.theta -= self.alpha * nabla

def add_polynomial_features(x, power):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return x ** np.arange(1, power + 1)

def plot_polynomial_curve(x, y, lr):
    continuous_x = np.linspace(0, 8, 1000)
    y_hat = lr.predict_(add_polynomial_features(
        continuous_x, lr.theta.size - 1))

    plt.plot(continuous_x, y_hat, "y-")
    plt.plot(x, y, 'bo')
    plt.show()

def get_poly_cost(x, y, poly, alpha=0.001, n_cycle=1000, theta=None):
    theta = np.ones(poly + 1)
    poly_x = add_polynomial_features(x, poly)

    lr = MyLR(theta, alpha=alpha, n_cycle=n_cycle)
    lr.fit_(poly_x, y)

    cost = lr.cost_(poly_x, y)
    print(f"Poly {poly}: {cost} | {repr(lr.theta)}")
    return (lr, cost)


data = pd.read_csv("../resources/are_blue_pills_magics.csv")

x = np.array(data["Micrograms"])
y = np.array(data["Score"])

cost_lst = []
lr_lst = []

for i in range(2, 10):
    (lr, cost) = get_poly_cost(x, y, i, alpha=9*(pow(10, -i - 5)), n_cycle=600000)
    lr_lst.append(lr)
    cost_lst.append(cost)

for lr in lr_lst:
       plot_polynomial_curve(x, y, lr)

plt.bar(list(range(2, len(cost_lst) + 2)), cost_lst)
plt.show()
