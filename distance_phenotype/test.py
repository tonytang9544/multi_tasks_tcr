import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

b = np.array([[7,8,9], [10, 11, 12]])

# print(np.append(a, b))
# print(np.append(a, b, axis=0))
# print(a.T.dot(np.array([1, 1])))

# from corr_utils import plot_correlation

# plot_correlation(dist_array=np.array([[1,2,3],[1,3,4]]), TABLO_data=2)

c = np.array([0, 1, 2])#, np.nan, 3, np.divide(0, 0)])
print(c)
print(c[~(c == 0)])
print(np.array([i for i in range(len(c))])[~np.isnan(c)])

print(np.divide([0, 1], [0, 1]))