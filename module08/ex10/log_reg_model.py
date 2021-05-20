import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def data_spliter(x, y, proportion):
    delim = int(x.shape[0] * proportion)
    concatenated = np.concatenate((x, y), 1)
    np.random.shuffle(concatenated)
    x, y = concatenated[..., :-1], concatenated[..., -1:]
    return x[:delim], x[delim:], y[:delim], y[delim:]

class MyLR():
    def add_intercept(self, x):
        return np.c_[np.ones(x.shape[0]), x]

    def __init__(self, theta, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = theta

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_(self, x):
        return self.sigmoid(self.add_intercept(x) @ self.theta)

    def cost_(self, x, y, eps=1e-15):
        t = y_hat + eps
        return -(((y @ np.log(t)) + ((1 - y) @ np.log(1 - t))) / y.shape[0])

    def log_gradient(self, x, y):
        pre_cal = self.add_intercept(x)
        return (pre_cal.T @ (self.predict_(x) - y)) / y.shape[0]

    def fit_(self, x, y):
        pre_cal = self.add_intercept(x)
        for _ in range(self.max_iter):
            predict = 1 / (1 + np.exp(-(pre_cal @ self.theta)))
            nabla = (pre_cal.T @ (predict - y)) / y.shape[0]
            self.theta -= self.alpha * nabla

def isMyFavoriteZipCode(zipcode, my_favorite_zipcode):
    return 1.0 if zipcode == my_favorite_zipcode else 0.0

def map_category(y, i):
    return np.vectorize(isMyFavoriteZipCode)(y, i)

def model_training(i, thetas, y_train, x_train):
    lr = MyLR(np.ones(y_train.shape[0]), alpha=4e-5, n_cycle=22000)
    # lr = MyLR(thetas[i], alpha=4e-5, n_cycle=22000)
    lr.fit_(x_train, map_category(y_train, i))
    return lr

y_data = pd.read_csv("../resources/solar_system_census_planets.csv", index_col=0)
x_data = pd.read_csv("../resources/solar_system_census.csv", index_col=0)
y = np.array(y_data)
x = np.array(x_data)

thetas = ([[ 4.90348242], [-0.02999681], [-0.03250215], [ 2.40047782]],
              [[ 1.28802845], [-0.06179008], [ 0.01894334], [ 8.00000601]],
              [[-4.75798975], [-0.00574743], [ 0.09731946], [-4.55362614]],
              [[-2.20593027], [ 0.08724529], [-0.09877385], [-8.59898021]])

x_train, x_test, y_train, y_test = data_spliter(x, y, 0.7)
trained_model = [model_training(i, thetas, y_train, x_train) for i in range(0, 4)]

predict = [model.predict_(x_data) for model in trained_model]

y_predict = np.argmax(np.concatenate(predict, axis=1), axis=1).reshape(-1, 1)

# compare = np.concatenate((y_test, y_predict), axis=1)
# compare = pd.DataFrame(compare.astype("int64"))
# print(pd.DataFrame(compare.astype("int64")))

unique, counts = np.unique(y_predict == y_test, return_counts=True)
print(dict(zip(unique, counts)))
