import numpy as np

def add_intercept(x):
    return np.c_[np.ones(x.shape[0]), x]

def logistic_predict(x, theta):
    return 1 / (1 + np.exp(-(add_intercept(x) @ theta)))

# Example 1
x = np.array([4])
theta = np.array([[2], [0.5]])
print(logistic_predict(x, theta))
# Output:
# array([[0.98201379]])

# Example 1
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print(logistic_predict(x2, theta2))
# Output:
# array([[0.98201379],
#        [0.99624161],
#        [0.97340301],
#        [0.99875204],
#        [0.90720705]])

# Example 3
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(logistic_predict(x3, theta3))
# Output:
# array([[0.03916572],
#        [0.00045262],
#        [0.2890505 ]])

