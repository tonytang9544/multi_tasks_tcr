import numpy as np
import pandas as pd
from tqdm import tqdm

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

dataset = pd.read_csv("20250814_1427_dataset_sub_sampled_10000.csv")
levenshtein_array = np.load("20250814_1427_levenshtein_array_10000.npy")

label_col = "label"
dataset[label_col] = dataset["annotation_L3"] == "MAIT"

max_distance = np.max(levenshtein_array)

correlation_array = np.zeros((2, max_distance+1))

sample_size, _ = levenshtein_array.shape

for i in tqdm(range(sample_size)):
    for j in range(i, sample_size):
        is_consistent = 0 if dataset.iloc[i][label_col] == dataset.iloc[j][label_col] else 1
        correlation_array[is_consistent, levenshtein_array[i, j]] += 1


print(correlation_array)
np.save(f"{running_time_stamp}_correlation_array", correlation_array)