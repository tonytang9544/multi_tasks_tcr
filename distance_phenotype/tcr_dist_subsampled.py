from pyrepseq.nn import nearest_neighbor_tcrdist
import pandas as pd
import numpy as np

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(f"script running time stamp is {running_time_stamp}")

sub_sample_size = 500000

dataset_path = "~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr.csv.gz"
dataset = pd.read_csv(dataset_path).sample(sub_sample_size)

print(dataset.head())


dataset.to_csv(f"{running_time_stamp}_dataset.csv.gz")
nn_array = nearest_neighbor_tcrdist(dataset, chain="both", max_edits=2, max_tcrdist=np.inf)
print(nn_array.shape)
np.save(f"{running_time_stamp}_nn_array", nn_array)