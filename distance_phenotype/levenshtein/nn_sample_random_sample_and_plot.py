from pyrepseq.nn import nearest_neighbor_tcrdist

from leven_utils import export_correlation_dict, sample_balanced_dataset, calculate_correlation_from_nn_samples, calculate_correlation_from_random_samples, plot_arrays

import pandas as pd
import os

import datetime

manual_logs = [
    f"Running script is:{"/".join(__file__.split("/")[-3:])}"
]
print(manual_logs[-1])

# configs
config_dict = {
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz",
    "nearest_neighbour_max_examples": 250000,
    "random_sample_examples": 2500,
    "annotation_level": "L1",
    "positive_phenotype_label": "CD4",
    "negative_phenotype_label": "CD8"
}

# record script start time
running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

manual_logs.append(f"script running time stamp is {running_time_stamp}")
print(manual_logs[-1])


save_path = f"./result/{running_time_stamp}"
manual_logs.append(f"result saving path is {save_path}")
print(manual_logs[-1])


if not os.path.exists(save_path):
    os.makedirs(save_path)

manual_logs.append(f"configuration dictionary: {config_dict}")
print(manual_logs[-1])

dataset_path = config_dict["dataset_path"]

manual_logs.append("Now loading the dataset.")
print(manual_logs[-1])

full_df = pd.read_csv(dataset_path).dropna()

_annotation_level = "annotation_" + config_dict["annotation_level"]
label_col="label"


###############
# sample two labels equally for nearest neighbour

manual_logs.append("Now create balanced dataset for nearest neighbour.")
print(manual_logs[-1])

dataset = sample_balanced_dataset(
    full_df=full_df,
    annotation_level=_annotation_level,
    positive_phenotype_label=config_dict["positive_phenotype_label"],
    negative_phenotype_label=config_dict["negative_phenotype_label"],
    nearest_neighbour_max_examples=config_dict["nearest_neighbour_max_examples"],
    dataset_export_path=os.path.join(save_path, "nn_sampled_dataset.csv.gz"),
    converted_label_col_name=label_col
)

################
# calculate levenshtein array

manual_logs.append("Now finding the nearest neighbours in the dataset using pyrepseq")
print(manual_logs[-1])

nn_array = nearest_neighbor_tcrdist(dataset, chain="both", max_edits=2, n_cpu=4)

manual_logs.append(f"Selected number of nearest neighbour pairs is {nn_array.shape[0]}")
print(manual_logs[-1])


#########
# calculate correlation for the pre-selected pairs

manual_logs.append("Now calculate the correlation from nearest neighbour samples")
print(manual_logs[-1])

levenshtein_phenotype_correlation_dict = calculate_correlation_from_nn_samples(
    nn_array=nn_array,
    dataset=dataset
)

levenshtein_phenotype_correlation_dict = dict(sorted(levenshtein_phenotype_correlation_dict.items()))


##############
# export nn array

nn_ratio_array = export_correlation_dict(
    levenshtein_phenotype_correlation_dict,
    os.path.join(save_path, "nn_sampled_results.csv.gz")
)



###############
# sample two labels equally for random sampling

manual_logs.append("Now generating balanced dataset using random sampling.")
print(manual_logs[-1])

dataset = sample_balanced_dataset(
    full_df=full_df,
    annotation_level=_annotation_level,
    positive_phenotype_label=config_dict["positive_phenotype_label"],
    negative_phenotype_label=config_dict["negative_phenotype_label"],
    nearest_neighbour_max_examples=config_dict["random_sample_examples"],
    dataset_export_path=os.path.join(save_path, "random_sampled_dataset.csv.gz"),
    converted_label_col_name=label_col
)


#################
# calculate levenshtein array
manual_logs.append("Now calculate the correlation from random samples")
print(manual_logs[-1])

random_sample_correlation = calculate_correlation_from_random_samples(
    dataset=dataset,
)

random_sample_correlation = dict(sorted(random_sample_correlation.items()))

random_ratio_array = export_correlation_dict(
    random_sample_correlation,
    os.path.join(save_path, "random_sampled_results.csv.gz")
)


###########
# save logs and save correlation arrays


with open(os.path.join(save_path, "run.log"), "w") as f:
    f.write("\n".join(manual_logs))


random_distances = list(random_sample_correlation.keys())


for each_dist in random_distances:
    if each_dist in levenshtein_phenotype_correlation_dict.keys():
        levenshtein_phenotype_correlation_dict[each_dist][0] += random_sample_correlation[each_dist][0]
        levenshtein_phenotype_correlation_dict[each_dist][1] += random_sample_correlation[each_dist][1]
    else:
        levenshtein_phenotype_correlation_dict[each_dist] = random_sample_correlation[each_dist].copy()

full_ratio_array = export_correlation_dict(
    levenshtein_phenotype_correlation_dict,
    os.path.join(save_path, "full_results.csv.gz")
)



##################
# plotting

plot_arrays(
    corr_arrays=[
        nn_ratio_array,
        random_ratio_array,
        full_ratio_array
    ],
    corr_plot_configs=[
        {"label": "nn_correlation",
         "color": "#800000",
         "marker": "s"},
        {"label": "random_correlation",
         "color": "#b00000",
         "marker": "^"},
        {"label": "combined_correlation",
         "color": "#f00000",
         "marker": "o"},
    ],
    sample_count_plot_configs=[
        {"label": "nn_example_counts",
         "color": "#505050",
         "marker": "s"},
        {"label": "random_example_counts",
         "color": "#909090",
         "marker": "^"},
        {"label": "combined_example_counts",
         "color": "#d0d0d0",
         "marker": "o"},
    ],
    fig_save_file=os.path.join(save_path, "edit_distance_phenotype.png")
)