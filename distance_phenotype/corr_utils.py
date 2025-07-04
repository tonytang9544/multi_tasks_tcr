import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_correlation(dist_array, TABLO_data, figure_name):
    '''
    plot correlation between CD4/CD8 consistency versus tcr_dist

    args:
        dist_array: 
            numpy ndarray of shape (number of tcr pairs, 3), 
            where each row is (index of tcr i, index of tcr j, tcr_dist between them)
        TABLO_data:
            pandas dataframe containing the original dataset,
            containing columns at least ["donor", "CD4_or_CD8", "TRAV", "TRAJ", "TRBV", "TRBJ", "CDR3A", "CDR3B"]
        figure_name:
            name of the figure to be saved onto disk
    '''

    # number of tcr pairs within a certain edit distance
    num_pairs, _ = dist_array.shape

    # maximum tcrdist found in the array
    max_tcrdist = np.max(dist_array[:, 2])
    print(max_tcrdist)

    # calculate distance correlation and store into an array
    # [[number of consistent pairs, number of inconsistent pairs, tcrdist] ...]
    dist_correlation_array = np.zeros((2, max_tcrdist+1))

    for i in range(num_pairs):
        tcr1, tcr2, tcr_dist = dist_array[i, :]
        if TABLO_data.iloc[tcr1]["CD4_or_CD8"] == TABLO_data.iloc[tcr2]["CD4_or_CD8"]:
            dist_correlation_array[0, tcr_dist] += 1
        else:
            dist_correlation_array[1, tcr_dist] += 1

    array_to_plot = np.zeros((max_tcrdist+1, 2))

    for i in range(max_tcrdist+1):
        total_counts_for_each_tcr_dist = dist_correlation_array[0, i*3] + dist_correlation_array[1, i*3]
        if total_counts_for_each_tcr_dist > 0:
            array_to_plot[i, :] = (i*3, dist_correlation_array[0, i*3] / total_counts_for_each_tcr_dist)

    print(array_to_plot)
    print(array_to_plot.shape)
    plt.plot(array_to_plot[:, 0], array_to_plot[:, 1])
    plt.xticks([i*3 for i in range(7)])
    plt.xlabel("TCR Dist")
    plt.ylabel("% same CD4/CD8 phenotypes")
    plt.savefig(figure_name)