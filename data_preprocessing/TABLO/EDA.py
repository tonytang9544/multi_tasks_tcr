# exploratory data analysis

import pandas as pd
import numpy as np


CD4_CD8_dataset = pd.read_csv('~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz')


print(CD4_CD8_dataset.head())
print(CD4_CD8_dataset.columns)
print(CD4_CD8_dataset["CD4_or_CD8"].value_counts())


# TABLO_dataset = pd.read_csv('~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz')
# TABLO_dataset["MAIT_or_not"] = TABLO_dataset["annotation_L3"] == "MAIT"
# print(TABLO_dataset["MAIT_or_not"].value_counts())


# annotation_cols = [f"annotation_L{i}" for i in range(1, 4)]

# for each_annotation in annotation_cols:
#     print(TABLO_dataset[each_annotation].unique())

# print(TABLO_dataset.shape[0])
# print(TABLO_dataset["full_id"].nunique())
# TABLO_dataset["tcr_id"] = TABLO_dataset["full_id"].apply(lambda x: x.split("_TR")[0])
# print(TABLO_dataset["tcr_id"].nunique())

# dataset = pd.read_csv("/home/minzhetang/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz")
# print(dataset.head())

# dataset = pd.read_csv("/home/minzhetang/Documents/dataset/TABLO_full_alpha_beta_combined.csv.gz")
# print(dataset.head())
# print(dataset.columns)

# annotation_columns = ['annotation_L1', 'annotation_L2', 'annotation_L3', 'annotation_L4', "cell_id", "umis_A", "umis_B"]
# print(dataset[annotation_columns].head())

# for col in annotation_columns:
#     print(f"for column: {col}")
#     print(dataset[col].unique())