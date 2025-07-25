from libtcrlm import schema
import pandas as pd

tcr_data_path = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset/CD4_CD8_sceptr.csv.gz"

dataset = pd.read_csv(tcr_data_path)

def generate_all_three_cdrs(dataset: pd.DataFrame):
    CDR1A = []
    CDR2A = []
    CDR1B = []
    CDR2B = []

    aa_seq_df = pd.DataFrame()

    for idx, entry in dataset.iterrows():
        tcr = schema.make_tcr_from_components(
            trav_symbol=entry["TRAV"],
            junction_a_sequence=entry["CDR3A"],
            trbv_symbol=entry["TRBV"],
            junction_b_sequence=entry["CDR3B"]
        )

        CDR1A.append(tcr.cdr1a_sequence)
        CDR2A.append(tcr.cdr2a_sequence)
        CDR1B.append(tcr.cdr1b_sequence)
        CDR2B.append(tcr.cdr2b_sequence)

    aa_seq_df["CDR1A"] = pd.Series(CDR1A)
    aa_seq_df["CDR2A"] = pd.Series(CDR2A)
    aa_seq_df["CDR1B"] = pd.Series(CDR1B)
    aa_seq_df["CDR2B"] = pd.Series(CDR2B)
    aa_seq_df["CDR3A"] = dataset["CDR3A"].copy()
    aa_seq_df["CDR3B"] = dataset["CDR3B"].copy()

    return aa_seq_df
    


dataset = generate_all_three_cdrs(dataset)

cdr_dataset_path = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset/CD4_CD8_sceptr_cdrs.csv.gz"
dataset.to_csv(cdr_dataset_path, index=False)