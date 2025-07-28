from Levenshtein import distance
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(f"script running time stamp is {running_time_stamp}")

dataset_path = "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz"

dataset = pd.read_csv(dataset_path).dropna()

pre_selected_dataset = pd.read_csv("/home/minzhetang/Documents/results/distance_phenotype/chunk_dataset/20250713/dataset_corresponding_to_chunk_0.csv.gz")
pre_selected_pairs = np.load("/home/minzhetang/Documents/results/distance_phenotype/chunk_dataset/20250713/nn_array_chunk_0.npy")

sub_sample_size = pre_selected_dataset.shape[0]

levenshtein_array = np.zeros((sub_sample_size, sub_sample_size), dtype=np.uint16)

cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]


for i in tqdm(range(sub_sample_size)):
    for j in range(i, sub_sample_size):
        total_distance = 0
        for cdr in cdrs:
            total_distance += distance(dataset.iloc[i][cdr], dataset.iloc[j][cdr])
            
        levenshtein_array[i, j] = total_distance
    # np.save(f"{running_time_stamp}_levenshtein_slice_{i}", levenshtein_array[i, :])

print(levenshtein_array)
np.save(f"{running_time_stamp}_levenshtein_array_{sub_sample_size}", levenshtein_array)