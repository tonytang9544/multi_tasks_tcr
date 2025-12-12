import numpy as np
import pandas as pd


# df = pd.read_csv(
#     "/home/minzhetang/Documents/results/distance_phenotype/20251203/20251203_1119_CD4_CD8/random_sampled_dataset.csv.gz", 
#     index_col=False
# )

# # print(df.head())
# # print(df.columns)

# df = df.sort_values(by="label", ascending=False)
# print(df.iloc[:25001]["label"].value_counts()[0])

# a = np.array([1, 2, 3, 4, 2, 1])

# print(np.unique(a, return_counts=True))

b = np.array([[2, 3, 4], [5, 2, 4]])#, [1, 1, 1]])

sub_sample_size = 3

# for i in range(sub_sample_size-1):
#     for j in range(i+1, sub_sample_size):
#         print(i, j)
b[[0, 1], 0] = np.array([10, 20])
print(b)
# print(str(None))

distances, counts = np.unique(
    b, 
    return_counts=True
)

print(distances, counts)
# df = df.rename(
#     columns={
#         'CDR1A': 'cdr1_a_aa',
#         'CDR1B': 'cdr1_b_aa',
#         'CDR2A': 'cdr2_a_aa',
#         'CDR2B': 'cdr2_b_aa',
#         'CDR3A': 'cdr3_a_aa',
#         'CDR3B': 'cdr3_b_aa',
#     }
# )
# df["pmhc_a_aa"] = ["A"] * df.shape[0]
# df["pmhc_b_aa"] = ["A"] * df.shape[0]



# tcr_rep = TCRrep(
#     cell_df=df, 
#     organism='human', 
#     chains=['alpha', 'beta'],
#     imgt_aligned      = False,
#     infer_all_genes   = False,
#     infer_cdrs        = False,
# )
# tcr_rep.compute_distances()


# print(tcr_rep.pw_alpha)
# print(tcr_rep.pw_beta)
