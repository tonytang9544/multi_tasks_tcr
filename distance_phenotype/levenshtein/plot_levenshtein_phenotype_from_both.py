import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(running_time_stamp)

corr_array = np.load("20250727_2324_correlation_array.npy")

total_count = corr_array.T.dot((1, 1))
print(total_count)

edit_dist_idx = np.array([i for i in range(corr_array.shape[1])])[~(total_count==0)]
ratio = corr_array[0, :][~(total_count==0)] / total_count[~(total_count==0)]

# with open(f"{running_time_stamp}_plot_data.pkl", "wb") as f:
#     pickle.dump(
#         {
#             "edit_dist_indices": edit_dist_idx,
#             "ratio": ratio
#         },
#         f
#     )

plt.scatter(edit_dist_idx, ratio)

agreement_dist = pd.read_csv("_edit_distance_phenotype_correlations_from_filtered_tcr_pairs.csv.gz")
edit_distance_indices = np.array([int(i) for i in agreement_dist.columns])
agreement_array = agreement_dist.to_numpy()
ratio = agreement_array[0] / agreement_array.dot((1, 1))
plt.scatter(edit_distance_indices, ratio)

plt.xlabel("Levenshtein distance")
plt.ylabel("CD4/CD8 phenotype correlations")
plt.title("phenotype agreement vs edit distance")
plt.savefig(f"{running_time_stamp}_edit_distance_phenotype.png")
plt.cla()
plt.close()