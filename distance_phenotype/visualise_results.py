from dist_corr_utils import plot_arrays
import os
import pandas as pd

source_path = "/home/minzhetang/Documents/results/distance_phenotype/CD4_CD8/balanced/result/20251010_1221"

plot_arrays(
    corr_arrays=[
        pd.read_csv(os.path.join(source_path, "nn_sampled_results.csv.gz")).to_numpy(),
        pd.read_csv(os.path.join(source_path, "random_sampled_results.csv.gz")).to_numpy(),
        pd.read_csv(os.path.join(source_path, "full_results.csv.gz")).to_numpy(),
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
    fig_save_file="edit_distance_phenotype.png"
)