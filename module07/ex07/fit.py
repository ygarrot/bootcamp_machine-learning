import numpy as np

class MyLR():
    def __init__(self, theta, alpha=5e-5, n_cycle=320000):
        self.theta = theta
        self.alpha = alpha
        self.n_cycle = n_cycle

    def add_intercept(self, x):
        return np.c_[np.ones(x.shape[0]), x]

    def predict(self, x):
        return self.add_intercept(x) @ self.theta

    def cost_elem_(self, x, y):
        dif = self.predict(x) - y
        return dif ** 2 / (2 * y.shape[0])

    def mse_(self, x, y):
        dif = (self.predict(x) - y).flatten()
        return dif @ dif / y.shape[0]

    def cost_(self, x, y):
        dif = (self.predict(x) - y).flatten()
        return dif @ dif / (2 * y.shape[0])

    def fit(self, x, y):
        pre_cal = self.add_intercept(x)
        for _ in range(self.n_cycle):
            nabla = (pre_cal.T @ ((pre_cal @ self.theta) - y)) / (y.shape[0])
            self.theta -= self.alpha * nabla


X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])
mylr = MyLR([[1.], [1.], [1.], [1.], [1.]])

# Example 0:
print(mylr.predict(X))
# Output:
# array([[8.], [48.], [323.]])

# Example 1:
print(mylr.cost_elem_(X,Y))
# Output:
# array([[37.5], [0.], [1837.5]])

# Example 2:
print(mylr.cost_(X,Y))
# Output:
# 1875.0

# Example 3:
print(mylr.fit(X, Y))
print(mylr.theta)
# Output:
# array([[18.023..], [3.323..], [-0.711..], [1.605..], [-0.1113..]])

# Example 4:
print(mylr.predict(X))
# Output:
# array([[23.499..], [47.385..], [218.079...]])

# Example 5:
print(mylr.cost_elem_(X,Y))
# Output:
# array([[0.041..], [0.062..], [0.001..]])

# Example 6:
print(mylr.cost_(X,Y))
# Output:
# 0.1056..

