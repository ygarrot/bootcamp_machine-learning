import numpy as np
def add_polynomial_features(x, power):
    s = x.shape
    r = (x.reshape(-1, 1) ** np.arange(2, power + 1))
    for e in r.T:
        x = np.concatenate((x, e.reshape(s)), axis=1)
    return x

x = np.arange(1,11).reshape(5, 2)

# print(x)
# Example 1:
print(add_polynomial_features(x, 3))
# Output:
# array([[   1,    2,    1,    4,    1,    8],
#        [   3,    4,    9,   16,   27,   64],
#        [   5,    6,   25,   36,  125,  216],
#        [   7,    8,   49,   64,  343,  512],
#        [   9,   10,   81,  100,  729, 1000]])

# Example 2:
print()
print(add_polynomial_features(x, 5))
# Output:
# array([[    1,     2,     1,     4,     1,     8,     1,    16],
#        [    3,     4,     9,    16,    27,    64,    81,   256],
#        [    5,     6,    25,    36,   125,   216,   625,  1296],
#        [    7,     8,    49,    64,   343,   512,  2401,  4096],
#        [    9,    10,    81,   100,   729,  1000,  6561, 10000]])

