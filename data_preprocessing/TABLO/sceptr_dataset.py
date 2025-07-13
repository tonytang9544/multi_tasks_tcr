import pandas as pd
import tidytcells

# load from file
col_renamed_df_path = "CD4_CD8_data_col_renamed.csv.gz"

dataset = pd.read_csv(col_renamed_df_path)
print(dataset.columns)

# select relavent columns
selected_cols = ["donor", "TRAV", "TRAJ", "CDR3A", "TRBV", "TRBJ", "CDR3B", "CD4_or_CD8"]

selected_df = dataset[selected_cols]
print(selected_df.head())

# new dataframe
sceptr_df = pd.DataFrame()

# pass on label and donor
pass_on_cols = ["donor", "CD4_or_CD8"]
for col in pass_on_cols:
    sceptr_df[col] = selected_df[col]

# standardise V, J
standardise_tcrs = ["TRAV", "TRAJ", "TRBV", "TRBJ"]

for col in standardise_tcrs:
    sceptr_df[col] = selected_df[col].apply(lambda x: tidytcells.tr.standardise(x, enforce_functional=True))

# standardise CDR3
standardise_junctions = ["CDR3A", "CDR3B"]

for col in standardise_junctions:
    sceptr_df[col] = selected_df[col].apply(lambda x: tidytcells.junction.standardise(x))

# drop NA for non-standard junctions
sceptr_df = sceptr_df.dropna()

# drop redundant tcrs
sceptr_df = sceptr_df.drop_duplicates()

print(sceptr_df.head())

# save to file
sceptr_df.to_csv("CD4_CD8_sceptr.csv.gz", index=False)