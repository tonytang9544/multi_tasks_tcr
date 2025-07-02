import pandas as pd
import tidytcells

col_renamed_df_path = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset/CD4_CD8_data_col_renamed.csv.gz"

dataset = pd.read_csv(col_renamed_df_path)
print(dataset.columns)

selected_cols = ["donor", "TRAV", "CDR3A", "TRBV", "CDR3B", "CD4_or_CD8"]

sceptr_df = dataset[selected_cols]

print(sceptr_df.head())

# standardise_cols = ["TRAV", "CDR3A", "TRBV", "CDR3B"]

# for col in standardise_cols:
#     sceptr_df[col].apply(lambda x: tidytcells.tr.standardise(x))

# print(sceptr_df.head())

sceptr_df_csv_path = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset/CD4_CD8_preprocessed.csv.gz"
sceptr_df.to_csv(sceptr_df_csv_path)