import numpy as np
import matplotlib.pyplot as plt

dist_correlation_array = np.load("/home/minzhetang/Documents/results/distance_phenotype/dist_correlation_array.npy")

array_to_plot = np.zeros((7, 2))

# array_to_plot[0, :] = (0, 1)
# print(array_to_plot)

for i in range(7):
    array_to_plot[i, :] = (i*3, dist_correlation_array[0, i*3] / (dist_correlation_array[0, i*3] + dist_correlation_array[1, i*3]))

print(array_to_plot)
print(array_to_plot.shape)
plt.plot(array_to_plot[:, 0], array_to_plot[:, 1])
plt.xticks([i*3 for i in range(7)])
plt.xlabel("TCR Dist")
plt.ylabel("% same CD4/CD8 phenotypes")
plt.savefig("correlation_plot.png")