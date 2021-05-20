import numpy as np

def add_intercept(x):
    return np.c_[np.ones(x.shape[0]), x]

class MyLR():
    def __init__(self, theta, alpha=0.001, n_cycle=1000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.theta = theta

    def predict_(self, x):
        return 1 / (1 + np.exp(-(add_intercept(x) @ self.theta)))

    def cost_(self, x, y, eps=1e-15):
        y = y.flatten()
        y_hat = self.predict_(x).flatten()
        return -(((y @ np.log(y_hat + eps)) + ((1 - y) @ np.log(1 - y_hat + eps))) / y.shape[0])

    def log_gradient(self, x, y):
        pre_cal = add_intercept(x)
        return (pre_cal.T @ (self.predict_(x) - y)) / y.shape[0]

    def fit_(self, x, y):
        pre_cal = add_intercept(x)
        y = y.flatten()
        for _ in range(self.max_iter):
            predict = 1 / (1 + np.exp(-(pre_cal @ self.theta)))
            nabla = (pre_cal.T @ (predict - y)) / y.shape[0]
            self.theta -= self.alpha * nabla

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09], n_cycle=22000)

# Example 0:
print(mylr.predict_(X))
# Output:
# array([[0.99930437],
#        [1.        ],
#        [1.        ]])

# Example 1:
print(mylr.cost_(X,Y))
# Output:
# 11.513157421577004

# Example 2:
mylr.fit_(X, Y)
print(mylr.theta)
# Output:
# array([[ 1.04565272],
#        [ 0.62555148],
#        [ 0.38387466],
#        [ 0.15622435],
#        [-0.45990099]])

# Example 3:
print(mylr.predict_(X))
# Output:
# array([[0.72865802],
#        [0.40550072],
#        [0.45241588]])

# Example 4:
print(mylr.cost_(X,Y))
# Output:
# 0.5432466580663214

