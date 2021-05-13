import numpy as np

def predict_(x, theta):
    return np.vstack((np.ones(len(x)), x)).T @ theta

def cost_(y, y_hat):
    sum = y_hat- y
    print(sum @ sum / (2 * y.shape[0]))
    # return np.dot(y_hat, y) * 2/y.shape[0]

import numpy as np
X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

# Example 1:
cost_(X, Y)
# Output:
# 2.142857142857143

# Example 2:
cost_(X, X)
# Output:
# 0.0