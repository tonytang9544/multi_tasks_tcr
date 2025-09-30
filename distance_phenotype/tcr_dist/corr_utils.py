import matplotlib.pyplot as plt
import numpy as np
import pickle


def plot_correlation_using_ndarray(dist_array, TABLO_data, figure_name="random_name"):
    '''
    calculate correlation between CD4/CD8 consistency versus tcr_dist

    args:
        dist_array: 
            numpy ndarray of shape (number of tcr pairs, 3), 
            where each row is (index of tcr i, index of tcr j, tcr_dist between them)
        TABLO_data:
            pandas dataframe containing the original dataset,
            containing columns at least ["donor", "CD4_or_CD8", "TRAV", "TRAJ", "TRBV", "TRBJ", "CDR3A", "CDR3B"]
        figure_name:
            name of the figure to be saved onto disk

    return:
    '''
    num_pairs, _ = dist_array.shape
    max_tcrdist = np.max(dist_array[:, 2])

    dist_correlation_array = np.zeros((2, max_tcrdist+1))

    for i in range(num_pairs):
        tcr1, tcr2, tcr_dist = dist_array[i, :]
        if TABLO_data.iloc[tcr1]["CD4_or_CD8"] == TABLO_data.iloc[tcr2]["CD4_or_CD8"]:
            dist_correlation_array[0, tcr_dist] += 1
        else:
            dist_correlation_array[1, tcr_dist] += 1
    
    # print(dist_correlation_array)
    np.save(f"dist_correlation_array_for_{figure_name}", dist_correlation_array)

    total = dist_correlation_array[0, :] + dist_correlation_array[1, :]

    tcr_dist_idx = np.array([i for i in range(max_tcrdist+1)])[~(total==0)]
    ratio = dist_correlation_array[0, :][~(total==0)] / total[~(total==0)]

    # plt.plot(tcr_dist_idx, ratio, label=figure_name)
    plt.scatter(tcr_dist_idx, ratio, label=figure_name)


def calculate_correlation_dict(dist_array, TABLO_data, figure_name="random_name"):
    '''
    calculate correlation between CD4/CD8 consistency versus tcr_dist

    args:
        dist_array: 
            numpy ndarray of shape (number of tcr pairs, 3), 
            where each row is (index of tcr i, index of tcr j, tcr_dist between them)
        TABLO_data:
            pandas dataframe containing the original dataset,
            containing columns at least ["donor", "CD4_or_CD8", "TRAV", "TRAJ", "TRBV", "TRBJ", "CDR3A", "CDR3B"]
        figure_name:
            name of the figure to be saved onto disk

    return:
        dictionary where
            key = tcr dist between tcr pair
            value = [number of tcr pairs that agrees on CD4/CD8 phenotype, number of those that do not agree]
    '''

    # number of tcr pairs within a certain edit distance
    num_pairs, _ = dist_array.shape

    # calculate distance correlation and store into a dictionary
    dist_correlation_dict = {}

    for i in range(num_pairs):
        tcr1, tcr2, tcr_dist = dist_array[i, :]
        if tcr_dist not in dist_correlation_dict.keys():
            dist_correlation_dict[tcr_dist] = [0, 0]
        if TABLO_data.iloc[tcr1]["CD4_or_CD8"] == TABLO_data.iloc[tcr2]["CD4_or_CD8"]:
            dist_correlation_dict[tcr_dist][0] += 1
        else:
            dist_correlation_dict[tcr_dist][1] += 1

    print(dist_correlation_dict)
    with open(f"dist_correlation_dict_for_{figure_name}.pkl", "wb") as f:
        pickle.dump(dist_correlation_dict, f)


def plot_correlation(dist_array=None, TABLO_data=None, dist_correlation_dict=None, figure_name="random_name"):
    if (dist_array is not None) and (TABLO_data is not None):
        dist_correlation_dict = calculate_correlation_dict(dist_array=dist_array, TABLO_data=TABLO_data, figure_name=figure_name)
    elif dist_correlation_dict is None:
        raise NotImplementedError("when dist_array and TABLO_data are not provided, dist_correlation_dict must be provided instead.")
    tcr_dists = [tcr_dist for tcr_dist in dist_correlation_dict.keys()]
    tcr_dists.sort()
    agreement_ratio = []

    for tcr_dist in tcr_dists:
        total_counts_for_each_tcr_dist = dist_correlation_dict[tcr_dist][0] + dist_correlation_dict[tcr_dist][1]
        agreement_ratio.append(dist_correlation_dict[tcr_dist][0] / total_counts_for_each_tcr_dist)

    plt.plot(tcr_dists, agreement_ratio, label=figure_name)
    # plt.xticks([i*3 for i in range(7)])
    # plt.xlabel("TCR Dist")
    # plt.ylabel("% same CD4/CD8 phenotypes")
    # plt.title(figure_name)
    # plt.savefig(figure_name)
