import numpy as np

def add_intercept(x):
    return np.c_[np.ones(x.shape[0]), x]

def logistic_predict(x, theta):
    return 1 / (1 + np.exp(-(add_intercept(x) @ theta)))


def vec_log_loss_(y, y_hat, eps=1e-15):
    y = y.flatten()
    return -((y @ np.log(y_hat) + (1 - y) @ np.log(1 - y_hat)) / y.shape[0])

def log_gradient(x, y, theta):
    pre_cal = add_intercept(x)
    return (pre_cal.T @ (logistic_predict(x, theta) - y)) / y.shape[0]
    


# Example 1:
y1 = np.array([1])
x1 = np.array([4])
theta1 = np.array([[2], [0.5]])

print(log_gradient(x1, y1, theta1))
# Output:
# array([[-0.01798621],
#        [-0.07194484]])

# Example 2:
y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])

print(log_gradient(x2, y2, theta2))
# Output:
# array([[0.3715235 ],
#        [3.25647547]])

# Example 3:
y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

print(log_gradient(x3, y3, theta3))
# Output:
# array([[-0.55711039],
#        [-0.90334809],
#        [-2.01756886],
#        [-2.10071291],
#        [-3.27257351]])
