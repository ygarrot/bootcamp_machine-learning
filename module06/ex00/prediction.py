import numpy as np

def predict_(x, theta):
    return np.vstack((np.ones(len(x)), x)).T @ theta

x = np.arange(1,6)

# Example 1:
theta1 = np.array([5, 0])
predict_(x, theta1)
# Ouput:
# array([5., 5., 5., 5., 5.])
theta2 = np.array([0, 1])
predict_(x, theta2)
theta3 = np.array([5, 3])
predict_(x, theta3)
theta4 = np.array([-3, 1])
predict_(x, theta4)