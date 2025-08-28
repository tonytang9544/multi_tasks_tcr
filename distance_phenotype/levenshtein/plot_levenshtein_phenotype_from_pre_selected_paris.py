import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import datetime

running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
print(running_time_stamp)

saved_counts = pd.read_csv("20250815_1054_edit_distance_phenotype_correlations_from_filtered_tcr_pairs.csv.gz")
edit_dist_idx = [int(i) for i in saved_counts.columns]
saved_counts_array = saved_counts.to_numpy()

total_count = saved_counts_array.T.dot((1, 1))
# print(edit_dist_idx)
# interesting_cols = [str(i) for i in range(6)]
# print(saved_counts)
# print(saved_counts[interesting_cols])
# # note how low the edit_distance=0 is! 
# # This is the case because all self-edit distance pairs are excluded (~ 500k). 
# # In addition, the duplicated columns were removed during dataset assembly, which means that edit distance 0 will never give the same phenotype!
# # If those are included, edit_distance=0 would approach 1.
# # Perhaps the good way to treat this problem is to ignore edit distance = 0? but this is actually very useful!

# print(saved_counts_array[0])

ratio = saved_counts_array[0] / total_count

# assembled = np.array([edit_dist_idx, ratio, total_count])



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
ax2.scatter(edit_dist_idx, total_count, color=color)
ax2.set_yscale("log")
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()

plt.savefig(f"{running_time_stamp}_edit_distance_phenotype_from_pre_selected_pairs.png")
plt.cla()
plt.close()