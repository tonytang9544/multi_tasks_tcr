# import numpy as np
# import pandas as pd

# from tcrdist.repertoire import TCRrep

# df = pd.read_csv(
#     "/home/minzhetang/Documents/results/distance_phenotype/CD4_CD8/balanced/result/20251103_1459/nn_sampled_dataset.csv.gz", 
#     index_col=False
# )

# print(df.head())
# print(df.columns)
print(str(None))

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
