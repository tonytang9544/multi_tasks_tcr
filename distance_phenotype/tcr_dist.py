from pyrepseq.nn import nearest_neighbor_tcrdist
import pandas as pd
import numpy as np
import os

dataset_root_path = "~/Documents/dataset"
dataset_path = os.path.join(dataset_root_path, "CD4_CD8_sceptr_nr_cdrs.csv.gz")
nn_array_path = os.path.join(dataset_root_path, "nn_array")
# nn_array_path = os.path.realpath(nn_array_path)
# print(nn_array_path)

dataset = pd.read_csv(dataset_path).sample(1000000)

print(dataset.head())

nn_array = nearest_neighbor_tcrdist(dataset, chain="both", max_edits=2)

print(nn_array)
print(len(nn_array))
np.save("nn_array", nn_array)