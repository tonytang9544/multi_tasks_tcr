from corr_utils import plot_correlation_using_ndarray
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load TABLO and calculated distance array
# dist_array = np.load("/home/minzhetang/Documents/results/distance_phenotype/semi_total_nn_array.npy")
# TABLO_data = pd.read_csv("/home/minzhetang/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz")


dist_arrays = {}
chunk_datasets = {}

for root, dirs, files in os.walk("/home/minzhetang/Documents/results/distance_phenotype/chunk_dataset/20250713"):
    for file in files:
        # print(file)
        if file.startswith("dataset_") and file.endswith(".csv.gz"):
            chunk_num = file.split(".csv")[0].split("chunk_")[1]
            chunk_datasets[chunk_num] = os.path.join(root, file)
        if file.startswith("nn_array") and file.endswith(".npy"):
            chunk_num = file.split(".npy")[0].split("chunk_")[1]
            dist_arrays[chunk_num] = os.path.join(root, file)

# print(chunk_datasets.keys())
# print(dist_arrays.keys())

# for i in range(6):
for i in range(1):
    dist_array = np.load(dist_arrays[str(i)])
    TABLO_data = pd.read_csv(chunk_datasets[str(i)])

    plot_correlation_using_ndarray(dist_array=dist_array, TABLO_data=TABLO_data, figure_name=f"chunk_{i}")

plt.xlabel("TCR Dist")
plt.ylabel("% same CD4/CD8 phenotypes")
plt.xticks([i*3 for i in range(7)])
plt.title("agreement ratio vs tcr dist")
plt.legend()
plt.savefig("correlation_plot_by_chunk")
plt.show()
    