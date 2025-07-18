import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load TABLO and calculated distance array
dist_array = np.load("/home/minzhetang/Documents/results/distance_phenotype/semi_total_nn_array.npy")
TABLO_data = pd.read_csv("/home/minzhetang/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz")

# number of tcr pairs within a certain edit distance
num_pairs, _ = dist_array.shape

# maximum tcrdist found in the array
max_tcrdist = np.max(dist_array[:, 2])
print(max_tcrdist)

# calculate distance correlation and store into an array
# [[number of consistent pairs, number of inconsistent pairs, tcrdist] ...]
dist_correlation_array = np.zeros((2, max_tcrdist+1))

for i in range(num_pairs):
    tcr1, tcr2, tcr_dist = dist_array[i, :]
    if TABLO_data.iloc[tcr1]["CD4_or_CD8"] == TABLO_data.iloc[tcr2]["CD4_or_CD8"]:
        dist_correlation_array[0, tcr_dist] += 1
    else:
        dist_correlation_array[1, tcr_dist] += 1


# the distance correlation array contains many tcr distances with zero entries, remove those for better plots
array_to_plot = np.zeros((7, 2))

for i in range(7):
    array_to_plot[i, :] = (i*3, dist_correlation_array[0, i*3] / (dist_correlation_array[0, i*3] + dist_correlation_array[1, i*3]))

print(array_to_plot)
print(array_to_plot.shape)
plt.plot(array_to_plot[:, 0], array_to_plot[:, 1])
plt.xticks([i*3 for i in range(7)])
plt.xlabel("TCR Dist")
plt.ylabel("% same CD4/CD8 phenotypes")
plt.savefig("correlation_plot.png")