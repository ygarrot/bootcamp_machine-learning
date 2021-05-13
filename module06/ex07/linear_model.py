import numpy as np
from ex06.my_linear_regression import MyLinearRegression as ML
from ex06.my_linear_regression import MyLinearRegression 
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.metrics import mean_squared_error

data = pd.read_csv("./resources/are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)

linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)
# plt.plot(Xpill, Yscore, '.')
# plt.plot(Xpill, Y_model1)
# plt.plot(Xpill, Y_model2)
theta0 = np.arange(80, 105, 5)
theta1 = np.linspace(-100, 100,  1000)

for i, t0 in enumerate(theta0):
    y_cost = [ ML(np.array([t0, t1])).mse_(Xpill, Yscore) for t1 in theta1]
    plt.plot(theta1, y_cost, label = "$J(\\theta_0 = C{}, \\theta_1)$".format(i))

ax = plt.gca()

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=3)

plt.xlabel("$\\theta_1$")
plt.ylabel("Cost function $J(\\theta_0,\\theta_1)$")
plt.axis([-14, -4, 0, 150])
plt.show()

print(linear_model1.mse_(Xpill, Yscore))
# 57.60304285714282
print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
print(linear_model2.mse_(Xpill, Yscore))
# 232.16344285714285
# error from the subject
# print(mean_squared_error(Yscore, Y_model1))
print(mean_squared_error(Yscore, Y_model2))
# 232.16344285714285
