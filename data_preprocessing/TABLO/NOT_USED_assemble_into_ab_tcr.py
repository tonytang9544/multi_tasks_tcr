import pandas as pd
import numpy as np

TABLO_dataset = pd.read_csv('~/Documents/dataset/TABLO_full_table_10donors.csv.gz')

print(TABLO_dataset.head())

TABLO_dataset["tcr_id"] = TABLO_dataset["full_id"].apply(lambda x: x.split("_TR")[0])

