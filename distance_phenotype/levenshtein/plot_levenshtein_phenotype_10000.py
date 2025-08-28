import matplotlib.pyplot as plt
import numpy as np
import pickle

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(running_time_stamp)

corr_array = np.load("20250814_2317_correlation_array.npy")

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

fig, ax1 = plt.subplots()

color = "blue"
ax1.set_xlabel("Levenshtein distance")
ax1.set_ylabel("CD4/CD8 phenotype correlations", color=color)
ax1.scatter(edit_dist_idx, ratio, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'red'
ax2.set_ylabel("total example counts", color=color) 
ax2.scatter(edit_dist_idx, total_count[~(total_count==0)], color=color)
ax2.set_yscale("log")
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("phenotype agreement vs edit distance")
plt.savefig(f"{running_time_stamp}_edit_distance_phenotype.png")
plt.cla()
plt.close()