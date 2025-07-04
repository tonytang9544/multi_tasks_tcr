import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# dist_correlation_dictionary = np.load("/home/minzhetang/Documents/results/distance_phenotype/dist_correlation_array.npy")

npy_list = []
for root, dirs, files in os.walk("/home/minzhetang/Documents/results/distance_phenotype/within_same_donor"):
    for file in files:
        if file.endswith(".npy"):
            npy_list.append(os.path.join(root, file))

print(npy_list)

TABLO_data = pd.read_csv("/home/minzhetang/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz")
# print(TABLO_data.head())



for npy in npy_list:
    array_to_plot = np.zeros((7, 2))
    dist_array = np.load(npy)
    donor_name = npy.split("_donor/")[1].split("_nn_array")[0]

    num_pairs, _ = dist_array.shape

    # print(dist_array[:, 2][:5])

    max_tcrdist = np.max(dist_array[:, 2])

    print(max_tcrdist)

    dist_correlation_array = np.zeros((2, max_tcrdist+1))

    for i in range(num_pairs):
        tcr1, tcr2, tcr_dist = dist_array[i, :]
        if TABLO_data.iloc[tcr1]["CD4_or_CD8"] == TABLO_data.iloc[tcr2]["CD4_or_CD8"]:
            dist_correlation_array[0, tcr_dist] += 1
        else:
            dist_correlation_array[1, tcr_dist] += 1
    
    print(dist_correlation_array)


    for i in range(7):
        array_to_plot[i, :] = (i*3, dist_correlation_array[0, i*3] / (dist_correlation_array[0, i*3] + dist_correlation_array[1, i*3]))

    print(array_to_plot)
    print(array_to_plot.shape)
    plt.plot(array_to_plot[:, 0], array_to_plot[:, 1], label=donor_name)
plt.xticks([i*3 for i in range(7)])
plt.xlabel("TCR Dist")
plt.ylabel("% same CD4/CD8 phenotypes")
plt.legend()
plt.savefig("within_donor_correlation_plot.png")