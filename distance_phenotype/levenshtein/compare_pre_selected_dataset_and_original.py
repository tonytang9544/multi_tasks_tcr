import pandas as pd

cdr_dataset = pd.read_csv("~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz").dropna()

pre_selected_dataset = pd.read_csv("/home/minzhetang/Documents/results/distance_phenotype/chunk_dataset/20250713/dataset_corresponding_to_chunk_0.csv.gz")

print(pre_selected_dataset.shape[0])
print(cdr_dataset["CDR3A"][:pre_selected_dataset.shape[0]])
print(pre_selected_dataset["CDR3A"])

# mask = cdr_dataset["CDR3A"][:pre_selected_dataset.shape[0]] == pre_selected_dataset["CDR3A"]
# print(mask)