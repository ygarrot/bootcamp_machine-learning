import numpy as np
import matplotlib.pyplot as plt
import functools

def decorator(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        value = func(*args, **kwargs)
        print(value)
        return value
    return wrapper_decorator

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

@decorator
def cost_elem_(y, y_hat):
    ret = (1/(2 * y.shape[0])) * ((y_hat - y)**2)
    return ret

@decorator
def cost_(y, y_hat):
    return cost_elem_(y, y_hat).sum()

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# Example 1:
cost_elem_(y1, y_hat1)
# Output:
# array([[0.], [0.1], [0.4], [0.9], [1.6]])

# Example 2:
cost_(y1, y_hat1)

# Output:
# 3.0

x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])

# Example 3:
cost_elem_(y2, y_hat2)

# Output:
# array([[1.3203125], [0.7503125], [0.0153125], [2.1528125]])

# Example 4:
cost_(y2, y_hat2)

# Output:
# 4.238750000000004

x3 = np.array([0, 15, -9, 7, 12, 3, -21])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])

# Example 5:
print("exemple 5")
print(y3)
print(y_hat3)

cost_(y3, y_hat3)

# Output:
# 2.142857142857143

# Example 6:
cost_(y3, y3)

# Output:
# 0.0