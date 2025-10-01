import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

levenshtein_phenotype_correlation_dict={
    "100": [2, 1],
    "10": [100, 3]
}

ratio_array = [[k, v[0] / (v[0] + v[1]), v[0]+v[1], v[0], v[1]] for k, v in levenshtein_phenotype_correlation_dict.items()]
ratio_array = np.array(ratio_array)


columns = [
    "edit_distance", 
    "corr_ratio", 
    "total_example_count", 
    "consistent_examples", 
    "inconsistent_examples"
]

result = pd.DataFrame(ratio_array, columns=columns)
result.to_csv("results_data.csv.gz", index=False)

print(pd.read_csv("results_data.csv.gz").head())

