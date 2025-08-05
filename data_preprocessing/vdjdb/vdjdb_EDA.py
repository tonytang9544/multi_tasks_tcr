import pandas as pd

VDJDB_dataset = pd.read_csv("~/Documents/dataset/20250610VDJDB.csv")

print(VDJDB_dataset["vbeta.gene"].head())