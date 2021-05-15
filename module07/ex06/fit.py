import numpy as np

def add_intercept(x):
    return np.c_[np.ones(x.shape[0]), x]

def predict_(x, theta):
    return np.c_[np.ones(theta.shape[0]), x] @ theta

def fit_(x, y, theta, alpha, n_cycle):
    x = add_intercept(x)
    for _ in range(n_cycle):
        nabla = (x.T @ ((x @ theta) - y)) / y.shape[0]
        theta -=  (alpha * nabla)
    return theta

import numpy as np
x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])

# Example 0:
theta2 = fit_(x, y, theta,  alpha = 0.0005, n_cycle=42000)
print(theta2)
# Output:
# array([[41.99..],[0.97..], [0.77..], [-1.20..]])

# Example 1:
print(predict_(x, theta2))
# Output:
# array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
