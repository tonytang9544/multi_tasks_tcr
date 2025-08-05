import pandas as pd
import tidytcells
from libtcrlm import schema
from tqdm import tqdm


mapping = {
    "valpha.id": "TRAV",
    "jalpha.id": "TRAJ",
    "cdr3alpha.id": "CDR3A",
    "vbeta.id": "TRBV",
    "jbeta.id": "TRBJ",
    "cdr3beta.id": "CDR3B",
}

tcr_data_path = "~/Documents/dataset/20250610VDJDB.csv"

print(f"********* loading file: {tcr_data_path}")
dataset = pd.read_csv(tcr_data_path)

print("********* renaming columns")
dataset = dataset.rename(columns=mapping)

# standardise V, J
print("********* standardising V and J genes")
standardise_tcrs = ["TRAV", "TRAJ", "TRBV", "TRBJ"]
for col in standardise_tcrs:
    dataset[col] = dataset[col].apply(lambda x: tidytcells.tr.standardise(x, enforce_functional=True))

# standardise CDR3
print("********* standardising CDR3")
standardise_junctions = ["CDR3A", "CDR3B"]
for col in standardise_junctions:
    dataset[col] = dataset[col].apply(lambda x: tidytcells.junction.standardise(x))

# drop NA for non-standard junctions and duplicates
print("********* dropping duplicates and na values")
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()

# map v genes to cdrs
print("********* mapping v genes to amino acid sequences")
CDR1A = []
CDR2A = []
CDR1B = []
CDR2B = []

for entry in tqdm(dataset.itertuples()):
    tcr = schema.make_tcr_from_components(
        trav_symbol=entry.TRAV,
        junction_a_sequence=entry.CDR3A,
        trbv_symbol=entry.TRBV,
        junction_b_sequence=entry.CDR3B
    )

    CDR1A.append(tcr.cdr1a_sequence)
    CDR2A.append(tcr.cdr2a_sequence)
    CDR1B.append(tcr.cdr1b_sequence)
    CDR2B.append(tcr.cdr2b_sequence)

dataset["CDR1A"] = pd.Series(CDR1A)
dataset["CDR2A"] = pd.Series(CDR2A)
dataset["CDR1B"] = pd.Series(CDR1B)
dataset["CDR2B"] = pd.Series(CDR2B)

print("********* finished preprocessing")
print("********* dataset summary is below")
print("dataset columns:")
print(dataset.columns)
print("dataset first 5 entries:")
print(dataset.head())

processed_dataset_path = "VDJDB_sceptr_nr_cdr.csv"
print(f"********* saving to disk: {processed_dataset_path}")
dataset.to_csv(processed_dataset_path, index=False)
