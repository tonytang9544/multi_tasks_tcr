import pandas as pd
import tidytcells

col_renamed_df_path = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset/CD4_CD8_data_col_renamed.csv.gz"

dataset = pd.read_csv(col_renamed_df_path)
print(dataset.columns)

selected_cols = ["donor", "TRAV", "TRAJ", "CDR3A", "TRBV", "TRBJ", "CDR3B", "CD4_or_CD8"]

selected_df = dataset[selected_cols]

print(selected_df.head())

sceptr_df = pd.DataFrame()

pass_on_cols = ["donor", "CD4_or_CD8"]

for col in pass_on_cols:
    sceptr_df[col] = selected_df[col]


standardise_tcrs = ["TRAV", "TRAJ", "TRBV", "TRBJ"]

for col in standardise_tcrs:
    sceptr_df[col] = selected_df[col].apply(lambda x: tidytcells.tr.standardise(x, enforce_functional=True))

standardise_junctions = ["CDR3A", "CDR3B"]

for col in standardise_junctions:
    sceptr_df[col] = selected_df[col].apply(lambda x: tidytcells.junction.standardise(x))

sceptr_df = sceptr_df.dropna()

print(sceptr_df.head())

sceptr_df_csv_path = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset/CD4_CD8_sceptr.csv.gz"
sceptr_df.to_csv(sceptr_df_csv_path, index=False)