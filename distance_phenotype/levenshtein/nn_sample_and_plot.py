from Levenshtein import distance
from pyrepseq.nn import nearest_neighbor_tcrdist
from tqdm import tqdm

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import datetime

# configs
config_dict = {
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz",
    "nearest_neighbour_max_examples": 100000,
    "annotation_level": "L1",
    "positive_phenotype_label": "CD4",
    "negative_phenotype_label": "CD8"
}

# record script start time
running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(f"script running time stamp is {running_time_stamp}")

save_path = f"./result/{running_time_stamp}"
print(f"result saving path is {save_path}")

if not os.path.exists(save_path):
    os.makedirs(save_path)

print(f"configuration dictionary: {config_dict}")

dataset_path = config_dict["dataset_path"]

full_df = pd.read_csv(dataset_path).dropna()

annotation_level = "annotation_" + config_dict["annotation_level"]

###############
# sample two labels equally

phenotype1_df = full_df[full_df[annotation_level] == config_dict["positive_phenotype_label"]].copy()

if "negative_phenotype_label" in config_dict.keys() and config_dict["negative_phenotype_label"] is not None:
    phenotype2_df = full_df[full_df[annotation_level] == config_dict["negative_phenotype_label"]].copy()
else:
    phenotype2_df = full_df[full_df[annotation_level] != config_dict["positive_phenotype_label"]].copy()

num_examples_per_label = min(phenotype1_df.shape[0], phenotype2_df.shape[0], config_dict["nearest_neighbour_max_examples"])
phenotype1_df = phenotype1_df.sample(num_examples_per_label)
phenotype2_df = phenotype2_df.sample(num_examples_per_label)

dataset = pd.concat([phenotype1_df, phenotype2_df])
dataset = dataset.sample(frac=1).reset_index(drop=True)

label_col = "label"
dataset[label_col] = dataset[annotation_level] == config_dict["positive_phenotype_label"]
dataset.to_csv(os.path.join(save_path, "dataset_sub_sampled.csv"))

################
# calculate levenshtein array

nn_array = nearest_neighbor_tcrdist(dataset, chain="both", max_edits=2)


#########
# calculate correlation

cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
levenshtein_phenotype_correlation_dict = {}

print("Now calculating correlation.")

for i in tqdm(range(nn_array.shape[0])):
    tcr1, tcr2, _ = nn_array[i, :]
    total_distance = 0
    for cdr in cdrs:
        total_distance += distance(dataset.iloc[tcr1][cdr], dataset.iloc[tcr2][cdr])

    is_consistent = 0 if dataset.iloc[tcr1][label_col] == dataset.iloc[tcr2][label_col] else 1

    if total_distance not in levenshtein_phenotype_correlation_dict.keys():
        levenshtein_phenotype_correlation_dict[total_distance] = [0, 0]
    levenshtein_phenotype_correlation_dict[total_distance][is_consistent] += 1

max_distance = max(levenshtein_phenotype_correlation_dict.keys())
correlation_array = np.zeros((2, max_distance+1))

for k, v in levenshtein_phenotype_correlation_dict.items():
    correlation_array[0, k] = levenshtein_phenotype_correlation_dict[k][0]
    correlation_array[1, k] = levenshtein_phenotype_correlation_dict[k][1]

print("calculated correlation array")
np.save(os.path.join(save_path, "correlation_array"), correlation_array)

total_count = correlation_array.T.dot((1, 1))
print("counted number of examples for each distance")

edit_dist_idx = np.array([i for i in range(correlation_array.shape[1])])[~(total_count==0)]
ratio = correlation_array[0, :][~(total_count==0)] / total_count[~(total_count==0)]


##################
# plotting

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
plt.savefig(os.path.join(save_path, "edit_distance_phenotype.png"))
plt.cla()
plt.close()