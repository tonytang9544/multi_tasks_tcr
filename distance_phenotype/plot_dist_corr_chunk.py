from corr_utils import plot_correlation
import os
import pandas as pd
import numpy as np

# load TABLO and calculated distance array
# dist_array = np.load("/home/minzhetang/Documents/results/distance_phenotype/semi_total_nn_array.npy")
# TABLO_data = pd.read_csv("/home/minzhetang/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz")


dist_arrays = {}
chunk_datasets = {}

for root, dirs, files in os.walk("../../../results/distance_phenotype/chunk_dataset"):
    for file in files:
        if file.endswith(".csv.gz"):
            chunk_num = file.split(".csv")[0].split("chunk_")[1]
            chunk_datasets[chunk_num] = os.path.join(root, file)
        if file.endswith(".npy"):
            chunk_num = file.split(".npy")[0].split("chunk_")[1]
            dist_arrays[chunk_num] = os.path.join(root, file)

# print(chunk_datasets.keys())
# print(dist_arrays.keys())

for i in range(6):
    dist_array = np.load(dist_arrays[str(i)])
    TABLO_data = pd.read_csv(chunk_datasets[str(i)])

    plot_correlation(dist_array, TABLO_data, f"correlation_plot_chunk_{i}")
    