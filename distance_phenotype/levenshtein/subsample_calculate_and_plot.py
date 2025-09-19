from Levenshtein import distance
from tqdm import tqdm

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import datetime

# configs
config_dict = {
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz",
    "sub_sample_size_each_phenotype": 5000,
    "annotation_level": "L1",
    "phenotype_label": "CD4",
    "negative_phenotype_label": "CD8"
}

# record script start time
running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(f"script running time stamp is {running_time_stamp}")

print(f"configuration dictionary: {config_dict}")

dataset_path = config_dict["dataset_path"]
sub_sample_size = config_dict["sub_sample_size_each_phenotype"]


full_df = pd.read_csv(dataset_path).dropna()

annotation_level = "annotation_" + config_dict["annotation_level"]

phenotype1_df = full_df[full_df[annotation_level] == config_dict["phenotype_label"]].sample(sub_sample_size).copy()

if "negative_phenotype_label" in config_dict.keys():
    phenotype2_df = full_df[full_df[annotation_level] == config_dict["negative_phenotype_label"]].sample(sub_sample_size).copy()
else:
    phenotype2_df = full_df[full_df[annotation_level] != config_dict["phenotype_label"]].sample(sub_sample_size).copy()

dataset = pd.concat([phenotype1_df, phenotype2_df])
dataset = dataset.sample(frac=1).reset_index(drop=True)

label_col = "label"
dataset[label_col] = dataset[annotation_level] == config_dict["phenotype_label"]
dataset.to_csv(f"{running_time_stamp}_dataset_sub_sampled.csv")

levenshtein_array = np.zeros((sub_sample_size, sub_sample_size), dtype=np.uint16)

cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]


for i in range(sub_sample_size):
    for j in range(i, sub_sample_size):
        total_distance = 0
        for cdr in cdrs:
            total_distance += distance(dataset.iloc[i][cdr], dataset.iloc[j][cdr])
            
        levenshtein_array[i, j] = total_distance
    # np.save(f"{running_time_stamp}_levenshtein_slice_{i}", levenshtein_array[i, :])

print(levenshtein_array)
np.save(f"{running_time_stamp}_levenshtein_array", levenshtein_array)

max_distance = np.max(levenshtein_array)

correlation_array = np.zeros((2, max_distance+1))

sample_size, _ = levenshtein_array.shape

for i in range(sample_size):
    for j in range(i, sample_size):
        is_consistent = 0 if dataset.iloc[i][label_col] == dataset.iloc[j][label_col] else 1
        correlation_array[is_consistent, levenshtein_array[i, j]] += 1


print(correlation_array)
np.save(f"{running_time_stamp}_correlation_array", correlation_array)

total_count = correlation_array.T.dot((1, 1))
print(total_count)

edit_dist_idx = np.array([i for i in range(correlation_array.shape[1])])[~(total_count==0)]
ratio = correlation_array[0, :][~(total_count==0)] / total_count[~(total_count==0)]

# with open(f"{running_time_stamp}_plot_data.pkl", "wb") as f:
#     pickle.dump(
#         {
#             "edit_dist_indices": edit_dist_idx,
#             "ratio": ratio
#         },
#         f
#     )

fig, ax1 = plt.subplots()

color = "blue"
ax1.set_xlabel("Levenshtein distance")
ax1.set_ylabel("phenotype correlations", color=color)
ax1.scatter(edit_dist_idx, ratio, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'red'
ax2.set_ylabel("total example counts", color=color) 
ax2.scatter(edit_dist_idx, total_count[~(total_count==0)], color=color)
ax2.set_yscale("log")
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("phenotype agreement vs edit distance")
plt.savefig(f"{running_time_stamp}_edit_distance_phenotype.png")
plt.cla()
plt.close()