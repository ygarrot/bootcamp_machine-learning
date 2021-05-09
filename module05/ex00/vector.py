class Vector():
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), 1)

    def foreach(self, callback, value):
        self.values = [[callback(val, value)] for lst in self.values for val in lst]
        return self.values

    def __add__(self, value):
        add = lambda a, b: a + b
        return self.foreach(add, value)

    def __radd__(self, value):
        return self.__add__(value)

v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
print(2+ v1)
