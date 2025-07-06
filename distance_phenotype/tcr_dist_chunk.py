from pyrepseq.nn import nearest_neighbor_tcrdist
import pandas as pd
import numpy as np

dataset_path = "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz"
dataset = pd.read_csv(dataset_path)

print(dataset.head())

sub_sample_size = 500000

for i in range(6):
    chunk_dataset = dataset[i*sub_sample_size: (i+1)*sub_sample_size]
    chunk_dataset.to_csv(f"dataset_corresponding_to_chunk_{i}.csv.gz")
    nn_array = nearest_neighbor_tcrdist(chunk_dataset, chain="both", max_edits=2, max_tcrdist=np.inf)
    print(nn_array.shape)
    np.save(f"nn_array_chunk_{i}", nn_array)