import numpy as np
import matplotlib
matplotlib.use('wxAgg') # no UI backend
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

def cost_elem_(y, y_hat):
    ret = (1/(2 * y.shape[0])) * ((y_hat - y)**2)
    return ret

def cost_(y, y_hat):
    return cost_elem_(y, y_hat).sum()

def plot_with_cost(x, y, theta):
    plt.plot(cost_elem_(y, predict(x, theta)))
    plt.show()


x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

# Example 1:
theta1= np.array([18,-1])
plot_with_cost(x, y, theta1)
# Output:

# Example 2:
theta2 = np.array([14, 0])
plot_with_cost(x, y, theta2)
# Output:

# Example 3:
theta3 = np.array([12, 0.8])
plot_with_cost(x, y, theta3)
# Output: