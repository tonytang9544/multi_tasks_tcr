import pandas as pd

VDJDB_dataset = pd.read_csv("~/Documents/dataset/20250610VDJDB.csv")

print(VDJDB_dataset.columns)

useful_cols = ["cdr3alpha.id", "cdr3beta.id", "epitope.id"]
print(VDJDB_dataset[useful_cols].head(20))

len_cols = [f"{name.split(".id")[0]}_len" for name in useful_cols]
print(len_cols)

for i in range(len(useful_cols)):
    VDJDB_dataset[len_cols[i]] = VDJDB_dataset[useful_cols[i]].apply(lambda x: len(x))

print(VDJDB_dataset.head())
print(VDJDB_dataset.describe())