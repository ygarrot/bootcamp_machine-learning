import numpy as np

def sigmoid_(x):
    return 1 / (1 + np.exp(-x))


# Example 1:
x = np.array(-4)
print(sigmoid_(x))
# Output:
# array([0.01798620996209156])

# Example 2:
x = np.array(2)
print(sigmoid_(x))
# Output:
# array([0.8807970779778823])

# Example 3:
x = np.array([[-4], [2], [0]])
print(sigmoid_(x))
# Output:
# array([[0.01798620996209156], [0.8807970779778823], [0.5]])
