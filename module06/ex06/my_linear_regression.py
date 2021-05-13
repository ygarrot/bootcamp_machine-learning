import numpy as np

class MyLinearRegression():
    def __init__(self,  thetas, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.thetas = thetas

    def predict_(self, x):
        return np.c_[np.ones(x.shape[0]), x] @ self.thetas

    def cost_elem_(self, y, x):
        y_hat = self.predict_(x)
        y = y.reshape(x.shape)
        dif = y_hat - y
        return dif **2 / (2 * y.shape[0])

    def cost_(self, x, y):
        y = y.flatten()
        y_hat = self.predict_(x)
        sum = y_hat - y
        return sum @ sum / (2 * y.shape[0])

    def fit_(self, x, y):
        y = y.flatten()
        trans = np.c_[np.ones(x.shape[0]), x]
        for _ in range(self.max_iter):
            nabla = (trans.T @ ((trans @ self.thetas) - y)) / y.shape[0]
            self.thetas[0] -= self.alpha * nabla[0]
            self.thetas[1] -= self.alpha * nabla[1]

    def mse_(self, x, y):
        y = y.flatten()
        y_hat = self.predict_(x)
        y_hat = y_hat.flatten()
        sum = y_hat - y
        return sum @ sum / (y.shape[0])

# x = np.array([[12.4956442], [21.5007972], [
#              31.5527382], [48.9145838], [57.5088733]])
# y = np.array([[37.4013816], [36.1473236], [
#              45.7655287], [46.6793434], [59.5585554]])

# lr1 = MyLinearRegression([2, 0.7])

# # Example 0.0:
# print(lr1.predict_(x))
# # Output:
# # array([[10.74695094],
# #        [17.05055804],
# #        [24.08691674],
# #        [36.24020866],
# #        [42.25621131]])

# # Example 0.1:
# print(lr1.cost_elem_(y, lr1.predict_(x)))
# print("after")
# # Output:
# # array([[77.72116511],
# #        [49.33699664],
# #        [72.38621816],
# #        [37.29223426],
# #        [78.28360514]])

# # Example 0.2:
# print(lr1.cost_(lr1.predict_(x), y))
# # Output:
# # 315.0202193084312


# # Example 1.0:
# lr2 = MyLinearRegression([1, 1], 5e-8, 1500000)
# lr2.fit_(x, y)
# print(lr2.thetas)
# # Output:
# # array([[1.40709365],
# #        [1.1150909 ]])

# # Example 1.1:
# print(lr2.predict_(x))
# # Output:
# # array([[15.3408728 ],
# #        [25.38243697],
# #        [36.59126492],
# #        [55.95130097],
# #        [65.53471499]])

# # Example 1.2:
# print("ls")
# print(lr2.cost_elem_(y, lr1.predict_(x)))
# # Output:
# # array([[35.6749755 ],
# #        [ 4.14286023],
# #        [ 1.26440585],
# #        [29.30443042],
# #        [22.27765992]])

# # Example 1.3:
# print(lr2.cost_(lr1.predict_(x), y))
# # Output:
# # 92.66433192085971

# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error


# data = pd.read_csv("../resources/are_blue_pills_magics.csv")
# Xpill = np.array(data[Micrograms]).reshape(-1,1)
# Yscore = np.array(data[Score]).reshape(-1,1)

# linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
# linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
# Y_model1 = linear_model1.predict_(Xpill)
# Y_model2 = linear_model2.predict_(Xpill)

# print(linear_model1.mse_(Xpill, Yscore))
# # 57.60304285714282
# print(mean_squared_error(Yscore, Y_model1))
# # 57.603042857142825
# print(linear_model2.mse_(Xpill, Yscore))
# # 232.16344285714285
# print(mean_squared_error(Yscore, Y_model1))
# # 232.16344285714285
