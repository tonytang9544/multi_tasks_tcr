import pandas as pd
from tqdm import tqdm


def align_datasets(reference_dataset, dataset_with_dropping):
    drop_list = []

    j = 0
    for i in tqdm(range(dataset_with_dropping.shape[0])):
        if reference_dataset["CDR3A"].iloc[j] == dataset_with_dropping["CDR3A"].iloc[i] and \
             reference_dataset["CDR3B"].iloc[j] == dataset_with_dropping["CDR3B"].iloc[i]:
            j += 1
        else:
            drop_list.append(i)
    
    return drop_list


# for drop_item in drop_list.reverse():
#     new_selected_dataset = new_selected_dataset.drop(drop_item)

# new_selected_dataset = new_selected_dataset.reset_index(drop=True)

# print(pre_selected_dataset.shape[0])
# print(cdr_dataset["CDR3A"].iloc[:pre_selected_dataset.shape[0]])
# print(pre_selected_dataset["CDR3A"])     


if __name__ == "__main__":
    cdr_dataset = pd.read_csv("~/Documents/results/data_preprocessing/TABLO/CD4_CD8_sceptr_nr_cdrs.csv.gz").dropna().reset_index(drop=True)

    pre_selected_dataset = pd.read_csv("/home/minzhetang/Documents/results/distance_phenotype/chunk_dataset/20250713/dataset_corresponding_to_chunk_0.csv.gz", index_col=0).drop_duplicates().reset_index(drop=True)



    drop_list = align_datasets(cdr_dataset, pre_selected_dataset)
    print(drop_list)
    print(len(drop_list))

    new_selected_dataset = pre_selected_dataset.drop(drop_list).reset_index(drop=True)
    print(new_selected_dataset.iloc[5136:5142])
    print(cdr_dataset.iloc[5136:5142])