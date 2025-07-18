# exploratory data analysis

import pandas as pd
import numpy as np

TABLO_dataset = pd.read_csv('~/Documents/dataset/CD4_CD8_data.csv.gz')

cdr3_cols = ["cdr3_A", "cdr3_B"]

print(TABLO_dataset[cdr3_cols].head())
