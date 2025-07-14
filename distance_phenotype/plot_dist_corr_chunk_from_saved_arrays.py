import os
import numpy as np
import matplotlib.pyplot as plt


corr_arrays = {}

for root, dirs, files in os.walk("/home/minzhetang/Documents/results/distance_phenotype/chunk_dataset/20250713"):
    for file in files:
        if file.startswith("dist_correlation_array") and file.endswith(".npy"):
            chunk_num = file.split(".npy")[0].split("chunk_")[1]
            corr_arrays[chunk_num] = np.load(os.path.join(root, file))

print(corr_arrays)

for chunk_num, dist_correlation_array in corr_arrays.items():
    total = dist_correlation_array[0, :] + dist_correlation_array[1, :]

    tcr_dist_idx = np.array([i for i in range(dist_correlation_array.shape[1])])[~(total==0)]
    ratio = dist_correlation_array[0, :][~(total==0)] / total[~(total==0)]

    plt.scatter(tcr_dist_idx, ratio, label=f"chunk_{chunk_num}")

plt.xlabel("TCR Dist")
plt.ylabel("% same CD4/CD8 phenotypes")
# plt.xticks([i*3 for i in range(7)])
plt.title("agreement ratio vs tcr dist")
plt.legend()
plt.savefig("correlation_plot_by_chunk")
plt.show()
    