import matplotlib.pyplot as plt
import numpy as np
import pickle

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

corr_array = np.load("250727_1745_correlation_array.npy")

total_count = corr_array.T.dot((1, 1))
print(total_count)

edit_dist_idx = np.array([i for i in range(corr_array.shape[1])])[~(total_count==0)]
ratio = corr_array[0, :][~(total_count==0)] / total_count[~(total_count==0)]

with open(f"{running_time_stamp}_plot_data.pkl", "wb") as f:
    pickle.dump(
        {
            "edit_dist_indices": edit_dist_idx,
            "ratio": ratio
        }
    )

plt.scatter(edit_dist_idx, ratio)
plt.xlabel("Levenshtein distance")
plt.ylabel("CD4/CD8 phenotype correlations")
plt.title("phenotype agreement vs edit distance")
plt.savefig(f"{running_time_stamp}_edit_distance_phenotype.png")
plt.cla()
plt.close()