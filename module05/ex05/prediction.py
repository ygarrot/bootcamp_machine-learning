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
    return x


def predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
      x: has to be an numpy.ndarray, a vector of dimension m * 1.
      theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
      y_hat as a numpy.ndarray, a vector of dimension m * 1.
      None if x or theta are empty numpy.ndarray.
      None if x or theta dimensions are not appropriate.
    Raises:
      This function should not raise any Exceptions.
    """
    # print(add_intercept(x))
    ret = add_intercept(x) * theta
    print(ret.sum(axis=1))

x = np.arange(1,6)
# Example 1:
theta1 = np.array([5, 0])
predict(x, theta1)
# Ouput:
#array([5., 5., 5., 5., 5.])
# Do you understand why y_hat contains only 5's here?  


# Example 2:
theta2 = np.array([0, 1])
predict(x, theta2)
# Output:
# array([1., 2., 3., 4., 5.])
# Do you understand why y_hat == x here?  


# Example 3:
theta3 = np.array([5, 3])
predict(x, theta3)
# Output:
# array([ 8., 11., 14., 17., 20.])


# Example 4:
theta4 = np.array([-3, 1])
predict(x, theta4)
# Output:
# array([-2., -1.,  0.,  1.,  2.])