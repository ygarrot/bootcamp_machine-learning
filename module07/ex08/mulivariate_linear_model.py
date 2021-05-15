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

    def predict(self, x):
        return self.add_intercept(x) @ self.theta

    def cost_elem_(self, x, y):
        dif = self.predict(x) - y
        return dif ** 2 / (2 * y.shape[0])

    def mse_(self, x, y):
        dif = (self.predict(x) - y).flatten()
        return dif @ dif / y.shape[0]

    def cost_(self, x, y):
        dif = (self.predict(x) - y).flatten()
        return dif @ dif / (2 * y.shape[0])

    def fit_(self, x, y):
        pre_cal = self.add_intercept(x)
        y = y.flatten()
        for _ in range(self.n_cycle):
            nabla = (pre_cal.T @ ((pre_cal @ self.theta) - y)) / (y.shape[0])
            self.theta -= self.alpha * nabla

from matplotlib.legend_handler import HandlerLine2D


data = pd.read_csv("../resources/spacecraft_data.csv")
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']])

def multiplot(my_lrage, cat):
    age_x = X
    tm = data[[cat]]
    my_lrage.fit_(age_x, Y)
    pred = my_lrage.predict(age_x)
    plt.plot(tm, Y, '.', 'sell price')
    plt.plot(tm, pred, 'o', 'predicted sell price')
    plt.show()

def plot(cat):
    my_lrage = MyLR([1.0, 1.0], alpha = 1e-5, n_cycle = 6000)
    age_x = data[[cat]]
    my_lrage.fit_(age_x, Y)
    pred = my_lrage.predict(age_x)
    line1, _ = plt.plot(age_x, Y, '.', 'sell price')
    plt.plot(age_x, pred, '.', 'predicted sell price')
    plt.show()

# plot('Thrust_power')

my_lreg = MyLR([1.0, 1.0, 1.0, 1.0], alpha = 1e-7, n_cycle = 60000)

my_lreg.mse_(X,Y)
# Output:
# 144044.877...

multiplot(my_lreg, 'Age')
exit()
my_lreg.fit_(X,Y)
my_lreg.theta
# Output:
# array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

my_lreg.mse_(X,Y)
# Output:
# 586.896999...

# import pandas as pd
# import numpy as np
# form mylinearregression import MyLinearRegression as MyLR

# data = pd.read_csv("spacecraft_data.csv")
# X = np.array(data[['Age']])
# Y = np.array(data[['Sell_price']])
# myLR_age = MyLR([[1000.0], [-1.0]])
# myLR_age.fit_(X[:,0].reshape(-1,1), Y, alpha = 2.5e-5, n_cycle = 100000)

# RMSE_age = myLR_age.mse_(X[:,0].reshape(-1,1),Y)
#  print(RMSE_age)
# 57636.77729...

