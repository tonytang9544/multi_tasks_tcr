import pandas as pd
import numpy as np

# label_col = "MAIT_or_NOT"
label_col = "CD4_CD8"


tcr_data_path = "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz"

tc_df = pd.read_csv(tcr_data_path).dropna().reset_index(drop=True)#.iloc[:1000]
tc_df = tc_df[(tc_df["annotation_L1"] == "CD4") | (tc_df["annotation_L1"] == "CD8")]
tc_df[label_col] = tc_df["annotation_L1"] == "CD4"

# print(tc_df.columns)

v_gene_dict = {
    "TRAV": list(tc_df["TRAV"].unique()),
    "TRBV": list(tc_df["TRBV"].unique())
}

print(v_gene_dict)

v_gene_phenotype_count = {}

for a_or_b in v_gene_dict.keys():
    for each_v_gene in v_gene_dict[a_or_b]:
        total_v_gene_sample = tc_df[tc_df[a_or_b] == each_v_gene]
        positives = total_v_gene_sample[total_v_gene_sample[label_col] == True].shape[0]
        negatives = total_v_gene_sample[total_v_gene_sample[label_col] == False].shape[0]
        v_gene_phenotype_count[each_v_gene] = [positives, negatives]

print(v_gene_phenotype_count)
pd.DataFrame(v_gene_phenotype_count).to_csv("v_gene_phenotype_count.csv", index=False)