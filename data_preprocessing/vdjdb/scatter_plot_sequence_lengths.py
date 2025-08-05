import pandas as pd
import matplotlib.pyplot as plt

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(f"script running time stamp is {running_time_stamp}")

VDJDB_dataset = pd.read_csv("~/Documents/dataset/20250610VDJDB.csv")

print(VDJDB_dataset.columns)

vgene_cols = ['valpha.gene', 'valpha.allele']
print(VDJDB_dataset[vgene_cols].head())


useful_cols = ["cdr3alpha.id", "cdr3beta.id", "epitope.id"]
print(VDJDB_dataset[useful_cols].head())

len_cols = [f"{name.split(".id")[0]}_len" for name in useful_cols]
# print(len_cols)

for i in range(len(useful_cols)):
    VDJDB_dataset[len_cols[i]] = VDJDB_dataset[useful_cols[i]].apply(lambda x: len(x))

# print(VDJDB_dataset.head())
# print(VDJDB_dataset.describe())

# print(VDJDB_dataset[VDJDB_dataset["cdr3alpha_len"] <= 7][useful_cols])

nr_dataset = VDJDB_dataset.drop_duplicates().copy()


for i in range(len(useful_cols)):
    nr_dataset[len_cols[i]] = nr_dataset[useful_cols[i]].apply(lambda x: len(x))

print(nr_dataset.describe())

lengths = [i for i in range(20)]

for col in len_cols:
    plt.scatter(lengths, [nr_dataset[nr_dataset[col] == length].shape[0] for length in lengths], label=col)
plt.xlabel("sequence length")
plt.xticks(lengths)
plt.ylabel("counts")
plt.yscale("log")
plt.legend()
plt.savefig(f"{running_time_stamp}_vdjdb_sequence_length.png")
plt.cla()
plt.close()
