import numpy as np

def l2(theta):
    return theta[1:] @ theta[1:]

def reg_log_cost_(y, y_hat, theta, lambda_):
    m = y.shape[0]
    r = (lambda_ / (2 * m)) * l2(theta)
    return -((y @ np.log(y_hat) + (1 - y) @ np.log(1 - y_hat)) / m) + r
    

y = np.array([1, 1, 0, 0, 1, 1, 0])
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01])
theta = np.array([1, 2.5, 1.5, -0.9])

# Example :
print(reg_log_cost_(y, y_hat, theta, .5))
# Output:
# 0.43377043716475955

# Example :
print(reg_log_cost_(y, y_hat, theta, .05))
# Output:
# 0.13452043716475953

# Example :
print(reg_log_cost_(y, y_hat, theta, .9))
# Output:
# 0.6997704371647596
