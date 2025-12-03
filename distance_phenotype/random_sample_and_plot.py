from pyrepseq.nn import nearest_neighbor_tcrdist

from dist_corr_utils import export_correlation_dict, sample_balanced_dataset, calculate_correlation_from_nn_samples, calculate_correlation_from_random_samples, plot_arrays

import pandas as pd
import os
import shutil

import datetime

manual_logs = [
    f"Running script is:{"/".join(__file__.split("/")[-3:])}"
]
print(manual_logs[-1])

# configs
config_dict = {
    "dataset_path": "~/Documents/results/data_preprocessing/TABLO/TABLO_full_sceptr_nr_cdr.csv.gz",
    "random_sample_examples": 25000,
    "annotation_level": "L3",
    "positive_phenotype_label": "MAIT",
    "negative_phenotype_label": None,
}

# record script start time
running_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

manual_logs.append(f"script running time stamp is {running_time_stamp}")
print(manual_logs[-1])


# make directories to save results

save_path = f"./{running_time_stamp}_{config_dict["positive_phenotype_label"]}_{str(config_dict["negative_phenotype_label"])}"
manual_logs.append(f"result saving path is {save_path}")
print(manual_logs[-1])


if not os.path.exists(save_path):
    os.makedirs(save_path)

manual_logs.append(f"configuration dictionary: {config_dict}")
print(manual_logs[-1])


# save driver script to the result folder for reproducibility

shutil.copy2(__file__, save_path)


# load dataset

dataset_path = config_dict["dataset_path"]

manual_logs.append("Now loading the dataset.")
print(manual_logs[-1])

full_df = pd.read_csv(dataset_path).dropna()

_annotation_level = "annotation_" + config_dict["annotation_level"]
label_col="label"

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



##################
# plotting

plot_arrays(
    corr_arrays=[
        random_ratio_array,
    ],
    corr_plot_configs=[
        {"label": "random_correlation",
         "color": "#b00000",
         "marker": "o"},
    ],
    sample_count_plot_configs=[
        {"label": "random_example_counts",
         "color": "#909090",
         "marker": "."},
    ],
    fig_save_file=os.path.join(save_path, "edit_distance_phenotype.png")
)