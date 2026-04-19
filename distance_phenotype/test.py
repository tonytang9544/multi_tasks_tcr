import numpy as np
import pandas as pd

distances = np.array([[[-1, 1, 2],
                      [1, -1, 1],
                      [2, 1, -1]],
                        [[-1, 0, 1],
                        [0, -1, 2],
                        [1, 2, -1]]],
                      dtype=np.int8)


consistent_distances_part1, consistent_counts_part1 = np.unique(
        distances, 
        return_counts=True
)

# Filter out self-comparisons
valid_indices = consistent_distances_part1 >= 0
consistent_distances_part1 = consistent_distances_part1[valid_indices]
consistent_counts_part1 = consistent_counts_part1[valid_indices] / 2

print(consistent_distances_part1)
print(consistent_counts_part1)
