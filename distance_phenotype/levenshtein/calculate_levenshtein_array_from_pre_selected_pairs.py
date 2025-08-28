from Levenshtein import distance
from libtcrlm import schema
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(f"script running time stamp is {running_time_stamp}")

# dataset_path = "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz"

# dataset = pd.read_csv(dataset_path).dropna().reset_index(drop=True)


def generate_all_three_cdrs(dataset: pd.DataFrame):
    CDR1A = []
    CDR2A = []
    CDR1B = []
    CDR2B = []

    aa_seq_df = dataset.copy()
    print("now loop through the dataset to generate all three cdrs")

    for idx, entry in tqdm(dataset.iterrows()):
        tcr = schema.make_tcr_from_components(
            trav_symbol=entry["TRAV"],
            junction_a_sequence=entry["CDR3A"],
            trbv_symbol=entry["TRBV"],
            junction_b_sequence=entry["CDR3B"]
        )

        CDR1A.append(tcr.cdr1a_sequence)
        CDR2A.append(tcr.cdr2a_sequence)
        CDR1B.append(tcr.cdr1b_sequence)
        CDR2B.append(tcr.cdr2b_sequence)

    aa_seq_df["CDR1A"] = pd.Series(CDR1A)
    aa_seq_df["CDR2A"] = pd.Series(CDR2A)
    aa_seq_df["CDR1B"] = pd.Series(CDR1B)
    aa_seq_df["CDR2B"] = pd.Series(CDR2B)

    return aa_seq_df


pre_selected_dataset = pd.read_csv("20250814_2319_dataset.csv.gz")
# print(pre_selected_dataset.columns)
pre_selected_dataset = generate_all_three_cdrs(pre_selected_dataset)
pre_selected_pairs = np.load("20250814_2319_nn_array.npy")

label_col = "label"
pre_selected_dataset[label_col] = pre_selected_dataset["annotation_L3"] == "MAIT"

# print(pre_selected_pairs.shape)

cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
levenshtein_phenotype_correlation_dict = {}

for i in tqdm(range(pre_selected_pairs.shape[0])):
    tcr1, tcr2, _ = pre_selected_pairs[i, :]
    total_distance = 0
    for cdr in cdrs:
        total_distance += distance(pre_selected_dataset.iloc[tcr1][cdr], pre_selected_dataset.iloc[tcr2][cdr])

    is_consistent = 0 if pre_selected_dataset.iloc[tcr1][label_col] == pre_selected_dataset.iloc[tcr2][label_col] else 1

    if total_distance not in levenshtein_phenotype_correlation_dict.keys():
        levenshtein_phenotype_correlation_dict[total_distance] = [0, 0]
    levenshtein_phenotype_correlation_dict[total_distance][is_consistent] += 1

print(levenshtein_phenotype_correlation_dict)
pd.DataFrame(levenshtein_phenotype_correlation_dict).to_csv(f"{running_time_stamp}_edit_distance_phenotype_correlations_from_filtered_tcr_pairs.csv.gz", index=False)


    # np.save(f"{running_time_stamp}_levenshtein_slice_{i}", levenshtein_array[i, :])


# print(levenshtein_array)
# np.save(f"{running_time_stamp}_levenshtein_array_{sample_size}", levenshtein_array)