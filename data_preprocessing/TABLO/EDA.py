# exploratory data analysis

import pandas as pd
import numpy as np

TABLO_dataset = pd.read_csv('~/Documents/dataset/TABLO_full_table_10donors.csv.gz')

print(TABLO_dataset.head())

annotation_cols = [f"annotation_L{i}" for i in range(1, 4)]

for each_annotation in annotation_cols:
    print(TABLO_dataset[each_annotation].unique())

print(TABLO_dataset.shape[0])
print(TABLO_dataset["full_id"].nunique())
TABLO_dataset["tcr_id"] = TABLO_dataset["full_id"].apply(lambda x: x.split("_TR")[0])
print(TABLO_dataset["tcr_id"].nunique())