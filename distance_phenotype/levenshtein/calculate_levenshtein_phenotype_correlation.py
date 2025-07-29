import numpy as np
import pandas as pd
from tqdm import tqdm

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

dataset = pd.read_csv("20250727_1814_dataset_sub_sampled_10000.csv")
levenshtein_array = np.load("20250727_1814_levenshtein_array_10000.npy")

max_distance = np.max(levenshtein_array)

correlation_array = np.zeros((2, max_distance+1))

sample_size, _ = levenshtein_array.shape

for i in tqdm(range(sample_size)):
    for j in range(i, sample_size):
        is_consistent = 0 if dataset.iloc[i]["CD4_or_CD8"] == dataset.iloc[j]["CD4_or_CD8"] else 1
        correlation_array[is_consistent, levenshtein_array[i, j]] += 1


print(correlation_array)
np.save(f"{running_time_stamp}_correlation_array", correlation_array)