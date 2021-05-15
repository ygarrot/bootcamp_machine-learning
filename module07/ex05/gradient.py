import numpy as np

def gradient(x, y, theta):
    return (x.T @ (x @ theta - y)) / y.shape[0]



x = np.array([
	      [ -6,  -7,  -9],
        [ 13,  -2,  14],
        [ -7,  14,  -1],
        [ -8,  -4,   6],
        [ -5,  -9,   6],
        [  1,  -5,  11],
        [  9, -11,   8]])
y = np.array([2, 14, -13, 5, 12, 4, -19])
theta1 = np.array([3,0.5,-6])

# Example :
print(gradient(x, y, theta1))
# Output:
# array([ -37.35714286,  183.14285714, -393.        ])

# Example :
theta2 = np.array([0,0,0])
print(gradient(x, y, theta2))
# Output:
# array([  0.85714286,  23.28571429, -26.42857143])


