import numpy as np

def split_one(data, proportion):
    # print(data)
    np.random.shuffle(data)
    ratio = int(data.shape[0] * proportion)
    # print(data[ratio:])
    return ( data[:ratio], data[ratio:])

def data_spliter(x, y, proportion):
    return(split_one(x, proportion), split_one(y, proportion))

x1 = np.array([1, 42, 300, 10, 59])
y = np.array([0,1,0,1,0])

# Example 1:
print(data_spliter(x1, y, 0.8))
# Output:
# (array([  1,  59,  42, 300]), array([10]), array([0, 0, 0, 1]), array([1]))

# Example 2:
print(data_spliter(x1, y, 0.5))
# Output:
# (array([59, 10]), array([  1, 300,  42]), array([0, 1]), array([0, 1, 0]))

x2 = np.array([ [  1,  42],
                [300,  10],
                [ 59,   1],
                [300,  59],
                [ 10,  42]])
y = np.array([0,1,0,1,0])

# Example 3:
print(data_spliter(x2, y, 0.8))
# Output:
# (array([[ 10,  42],
#         [300,  59],
#         [ 59,   1],
#         [300,  10]]), array([[ 1, 42]]), array([0, 1, 0, 1]), array([0]))

# Example 4:
print(data_spliter(x2, y, 0.5))
# Output:
# (array([[59,  1],
#         [10, 42]]), array([[300,  10],
#         [300,  59],
#         [  1,  42]]), array([0, 0]), array([1, 1, 0]))

# Be careful! The way tuples of arrays are displayed could be a bit confusing... 
# 
# In the last example, the tuple returned contains the following arrays: 
# array([[59,  1],
#        [10, 42]])
#
# array([[300,  10],
#        [300,  59]
#
# array([0, 0])
#
# array([1, 1, 0]))

