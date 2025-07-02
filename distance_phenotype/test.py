import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

b = np.array([[7,8,9], [10, 11, 12]])

print(np.append(a, b))
print(np.append(a, b, axis=0))