import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from Levenshtein import distance

def export_correlation_dict(
    levenshtein_phenotype_correlation_dict: dict,
    export_file_name: str="result.csv.gz"
):

    ratio_array = [[k, v[0] / (v[0] + v[1]), v[0]+v[1], v[0], v[1]] 
                for k, v in levenshtein_phenotype_correlation_dict.items()]
    ratio_array = np.array(ratio_array)


    columns = [
        "edit_distance", 
        "corr_ratio", 
        "total_example_count", 
        "consistent_examples", 
        "inconsistent_examples"
    ]

    pd.DataFrame(
        ratio_array, 
        columns=columns
    ).to_csv(
        export_file_name, 
        index=False
    )

    return ratio_array


def sample_balanced_dataset(
    full_df: pd.DataFrame,
    annotation_level: str="annotation_L1",
    positive_phenotype_label: str="CD8",
    negative_phenotype_label: str="CD8",
    nearest_neighbour_max_examples: int=2500,
    dataset_export_path: str=None,
    converted_label_col_name: str="label"
):

    phenotype1_df = full_df[full_df[annotation_level] == positive_phenotype_label].copy()

    if negative_phenotype_label is not None:
        phenotype2_df = full_df[full_df[annotation_level] == negative_phenotype_label].copy()
    else:
        phenotype2_df = full_df[full_df[annotation_level] != positive_phenotype_label].copy()

    num_examples_per_label = min(phenotype1_df.shape[0], phenotype2_df.shape[0], nearest_neighbour_max_examples)
    phenotype1_df = phenotype1_df.sample(num_examples_per_label)
    phenotype2_df = phenotype2_df.sample(num_examples_per_label)

    dataset = pd.concat([phenotype1_df, phenotype2_df])
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    label_col = converted_label_col_name
    dataset[label_col] = dataset[annotation_level] == positive_phenotype_label
    if dataset_export_path is not None:
        dataset.to_csv(dataset_export_path)

    return dataset


def calculate_correlation_from_nn_samples(
    nn_array,
    dataset,
    converted_label_col_name: str="label",
    distance_function = distance
):
    '''
    calculate phenotype correlation for each pairs specified in nearest neighbour samples.

    input
    ----------
        nn_array: the nearest neighbour array provided from pyrepseq.nn.nearest_neighbor_tcrdist
        dataset: the dataset used to generate nearest neighbour array
        converted_label_col_name: name of the new column to be created to store positive/negative label for each sample
        distance_function: function of distance that takes the form (seq1, seq2) -> int/float

    output
    ----------
        dictionary of {distance: [number of consistent examples, number of inconsistent examples]}
    '''
    cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]
    levenshtein_phenotype_correlation_dict = {}

    for i in tqdm(range(nn_array.shape[0])):
        tcr1, tcr2, _ = nn_array[i, :]
        total_distance = 0
        for cdr in cdrs:
            total_distance += distance_function(dataset.iloc[tcr1][cdr], dataset.iloc[tcr2][cdr])

        is_consistent = 0 if dataset.iloc[tcr1][converted_label_col_name] == dataset.iloc[tcr2][converted_label_col_name] else 1

        if total_distance not in levenshtein_phenotype_correlation_dict.keys():
            levenshtein_phenotype_correlation_dict[total_distance] = [0, 0]
        levenshtein_phenotype_correlation_dict[total_distance][is_consistent] += 1

    return levenshtein_phenotype_correlation_dict


def calculate_correlation_from_random_samples(
    dataset,
    converted_label_col_name: str="label",
    distance_function = distance
):
    '''
    calculate phenotype correlation for each pairs specified in nearest neighbour samples.

    input
    ----------
        dataset: the dataset that contains the following columns:
            CDR1A
            CDR2A
            CDR3A
            CDR1B
            CDR2B
            CDR3B
            converted_label_col_name
        converted_label_col_name: name of the new column to be created to store positive/negative label for each sample
        distance_function: function of distance that takes the form (seq1, seq2) -> int/float

    output
    ----------
        dictionary of {distance: [number of consistent examples, number of inconsistent examples]}
    '''
    random_sample_correlation = {}

    cdrs = ["CDR1A", "CDR2A", "CDR3A", "CDR1B", "CDR2B", "CDR3B"]

    sub_sample_size = dataset.shape[0]

    for i in tqdm(range(sub_sample_size)):
        for j in range(i, sub_sample_size):
            total_distance = 0
            for cdr in cdrs:
                total_distance += distance_function(dataset.iloc[i][cdr], dataset.iloc[j][cdr])

            is_consistent = 0 if dataset.iloc[i][converted_label_col_name] == dataset.iloc[j][converted_label_col_name] else 1

            if total_distance not in random_sample_correlation.keys():
                random_sample_correlation[total_distance] = [0, 0]
            random_sample_correlation[total_distance][is_consistent] += 1

    return random_sample_correlation


def plot_arrays(
    corr_arrays: list,
    corr_plot_configs: list,
    sample_count_plot_configs: list,
    fig_save_file: str="edit_distance_phenotype.png"
):
    '''
    input
    ----------
        corr_arrays: 
            list of correlation arrays
        corr_plot_configs: 
            list of dictionaries containing configs for plotting correlations,
            must have same length as corr_arrays
        sample_count_plot_configs:
            list of dictionaries containing configs for plotting sample number counts,
            must have same length as corr_arrays
    '''

    assert len(corr_arrays) == len(corr_plot_configs), "corr_plot_configs must have the same number of elements as corr_arrays."
    assert len(corr_arrays) == len(sample_count_plot_configs), "sample_count_plot_configs must have the same number of elements as corr_arrays."

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Levenshtein distance")
    ax1.set_ylabel("phenotype correlations")
    for i in range(len(corr_arrays)):
        ax1.scatter(corr_arrays[i][:, 0], corr_arrays[i][:, 1], **corr_plot_configs[i])
    # ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    # color = 'red'
    ax2.set_ylabel("total example counts")
    for i in range(len(corr_arrays)):
        ax2.scatter(corr_arrays[i][:, 0], corr_arrays[i][:, 2], **sample_count_plot_configs[i])
    ax2.set_yscale("log")
    # ax2.tick_params(axis='y', labelcolor=color)

    # fig.tight_layout()
    ax1.legend()
    ax2.legend()
    # plt.legend()

    # plt.title("phenotype agreement vs edit distance")
    plt.savefig(fig_save_file)
    plt.cla()
    plt.close()