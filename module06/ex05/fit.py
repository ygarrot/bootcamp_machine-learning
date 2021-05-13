import numpy as np

def predict_(x, theta):
    return np.vstack((np.ones(x.shape[0]), x)).T @ theta

def cost(y, x, theta):
    trans = np.c_[np.ones(x.shape[0]), x]
    # can also be writen as 
    # trans = np.vstack((np.ones(x.shape[0]), x)).T
    return (trans.T @ ((trans @ theta) - y)) / y.shape[0]

def fit(x, y, theta, alpha, max_iter):
    y = y.flatten()
    trans = np.c_[np.ones(x.shape[0]), x]
    for _ in range(max_iter):
        nabla = (trans.T @ ((trans @ theta) - y)) / y.shape[0]
        theta[0] -= alpha * nabla[0]
        theta[1] -= alpha * nabla[1]

    return theta



x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1])

# Example 0:
theta1 = fit(x, y, theta, alpha=5e-8, max_iter=1500000)
# Output:
# array([[1.40709365],
#        [1.1150909 ]])

# Example 1:
print(theta1)
print(predict_(x.flatten(), theta1.flatten()))
# Output:
# array([[15.3408728 ],
#        [25.38243697],
#        [36.59126492],
#        [55.95130097],
#        [65.53471499]])