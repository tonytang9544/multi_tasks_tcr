import numpy as np
import pandas as pd

dist_array = np.load("/home/minzhetang/Documents/results/distance_phenotype/semi_total_nn_array.npy")
# print(dist_array.shape)

TABLO_data = pd.read_csv("/home/minzhetang/Documents/dataset/CD4_CD8_sceptr_nr_cdrs.csv.gz")
# print(TABLO_data.head())

num_pairs, _ = dist_array.shape

# print(dist_array[:, 2][:5])

max_tcrdist = np.max(dist_array[:, 2])

print(max_tcrdist)

dist_correlation_array = np.zeros((2, max_tcrdist+1))
dist_correlation_dictionary = {}

for i in range(num_pairs):
    tcr1, tcr2, tcr_dist = dist_array[i, :]
    if TABLO_data.iloc[tcr1]["CD4_or_CD8"] == TABLO_data.iloc[tcr2]["CD4_or_CD8"]:
        dist_correlation_array[0, tcr_dist] += 1
    else:
        dist_correlation_array[1, tcr_dist] += 1

print(dist_correlation_array)
np.save("dist_correlation_array", dist_correlation_array)
