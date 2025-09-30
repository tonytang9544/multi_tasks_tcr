from pyrepseq.nn import nearest_neighbor_tcrdist
import pandas as pd
import numpy as np
import os

dataset_root_path = "~/Documents/dataset"
dataset_path = os.path.join(dataset_root_path, "CD4_CD8_sceptr_nr_cdrs.csv.gz")
nn_array_path = os.path.join(dataset_root_path, "nn_array")
# nn_array_path = os.path.realpath(nn_array_path)
# print(nn_array_path)

dataset = pd.read_csv(dataset_path)

print(dataset.head())

sub_sample_size = 500000

total_nn_array = None

for i in range(6):
    nn_array = nearest_neighbor_tcrdist(dataset[i*sub_sample_size: (i+1)*sub_sample_size], chain="both", max_edits=2)
    print(nn_array.shape)
    if total_nn_array is None:
        total_nn_array = nn_array
    else:
        total_nn_array = np.append(total_nn_array, nn_array, axis=0)

print(total_nn_array.shape)
np.save("nn_array", total_nn_array)