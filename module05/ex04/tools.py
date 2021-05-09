import numpy as np

def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x.
    Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
    Returns:
      X as a numpy.ndarray, a vector of dimension m * 2.
      None if x is not a numpy.ndarray.
      None if x is a empty numpy.ndarray.
    Raises:
      This function should not raise any Exception.
    """
    if (len(x.shape)==1):
        x = x[:, np.newaxis]
    x =np.append(np.full((x.shape[0], 1), 1),x, axis=1)

    print(x)
# Example 1:
x = np.arange(1,6)
add_intercept(x)
# Output:
# array([[1., 1.],
#        [1., 2.],
#        [1., 3.],
#        [1., 4.],
#        [1., 5.]])


# Example 2:
y = np.arange(1,10).reshape((3,3))
add_intercept(y)
# Output:
# array([[1., 1., 2., 3.],
#        [1., 4., 5., 6.],
#        [1., 7., 8., 9.]])