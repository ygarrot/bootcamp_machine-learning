import numpy as np
from mylinearregression import MyLinearRegression as MyLR
x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

lr1 = MyLR([2, 0.7])

# Example 0.0:
print(lr1.predict(x))
# Output:
# array([[10.74695094],
#        [17.05055804],
#        [24.08691674],
#        [36.24020866],
#        [42.25621131]])

# Example 0.1:
print(lr1.cost_elem_(lr1.predict(x),y))
# Output:
# array([[77.72116511],
#        [49.33699664],
#        [72.38621816],
#        [37.29223426],
#        [78.28360514]])

# Example 0.2:
print(lr1.cost_(lr1.predict(x),y))
# Output:
# 315.0202193084312


# Example 1.0:
lr2 = MyLR([0, 0])
lr2.fit(x, y, alpha=5e-8, n_cycle = 1500000)
print(lr2.theta)
# Output:
# array([[1.40709365],
#        [1.1150909 ]])

# Example 1.1:
print(lr2.predict(x))
# Output:
# array([[15.3408728 ],
#        [25.38243697],
#        [36.59126492],
#        [55.95130097],
#        [65.53471499]])

# Example 1.2:
print(lr2.cost_elem_(lr1.predict(x),y))
# Output:
# array([[35.6749755 ],
#        [ 4.14286023],
#        [ 1.26440585],
#        [29.30443042],
#        [22.27765992]])

# Example 1.3:
print(lr2.cost_(lr1.predict(x),y))
# Output:
# 92.66433192085971

