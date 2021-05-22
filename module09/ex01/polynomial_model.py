import numpy as np

def add_polynomial_features(x, power):
    return x ** np.arange(1, power + 1)


x = np.arange(1,6).reshape(-1, 1)

print(x)
# Example 1:
print(add_polynomial_features(x, 3))
# Output:
# array([[  1,   1,   1],
#        [  2,   4,   8],
#        [  3,   9,  27],
#        [  4,  16,  64],
#        [  5,  25, 125]])


# Example 2:
print(add_polynomial_features(x, 6))
# Output:
# array([[    1,     1,     1,     1,     1,     1],
#        [    2,     4,     8,    16,    32,    64],
#        [    3,     9,    27,    81,   243,   729],
#        [    4,    16,    64,   256,  1024,  4096],
#        [    5,    25,   125,   625,  3125, 15625]])

