import numpy as np
import matplotlib.pyplot as plt

def add_intercept(x):
    if (len(x.shape)==1):
        x = x[:, np.newaxis]
    x = np.append(np.full((x.shape[0], 1), 1),x, axis=1)
    return x

def predict(x, theta):
    ret = theta.reshape((theta.shape[0],)) * add_intercept(x) 
    ret = ret.sum(axis=1)
    ret = ret.reshape((ret.shape[0], -1))
    return ret

def cost_(x, y):
    return ((1 / (2*len(y))) *((x - y)**2)).sum()

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

# Example 1:
print(cost_(X, Y))
# Output:
# 2.142857142857143

# Example 2:
print(cost_(X, X))
# Output:
# 0.0