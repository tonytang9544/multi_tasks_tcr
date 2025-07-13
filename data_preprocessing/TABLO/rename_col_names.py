import pandas as pd

if __name__ == "__main__":

    mapping = {
            "v_gene_A": "TRAV",
            "j_gene_A": "TRAJ",
            "cdr3_A": "CDR3A",
            "v_gene_B": "TRBV",
            "j_gene_B": "TRBJ",
            "cdr3_B": "CDR3B",
        }

    tcr_data_path = "~/Documents/dataset/CD4_CD8_data.csv.gz"

    dataset = pd.read_csv(tcr_data_path)
    col_rename_df = dataset.rename(columns=mapping)
    
    print(col_rename_df.columns)
    print(col_rename_df.head())
    col_rename_df.to_csv("CD4_CD8_data_col_renamed.csv.gz")