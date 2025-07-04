from pyrepseq.nn import nearest_neighbor_tcrdist
import pandas as pd
import numpy as np
import os

dataset_path = "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz"


dataset = pd.read_csv(dataset_path)

print(dataset.head())

donors = dataset["donor"].unique()

print(donors)

for each_donor in donors:
    print(dataset[dataset["donor"] == each_donor].shape)

for each_donor in donors:
    nn_array = nearest_neighbor_tcrdist(dataset[dataset["donor"] == each_donor], chain="both", max_edits=2)
    print(nn_array.shape)
    np.save(f"{each_donor}_nn_array", nn_array)
