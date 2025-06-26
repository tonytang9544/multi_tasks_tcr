from pyrepseq.nn import nearest_neighbor_tcrdist
import pandas as pd

dataset_path = "/Users/tangm/The Francis Crick Dropbox/Minzhe Tang/Tony/After-PhD/Machine_learning_MSc/UCL-AI_for_biomed/Course_material/thesis_project/TCR_project/dataset/CD4_CD8_sceptr_nr_cdrs.csv.gz"

dataset = pd.read_csv(dataset_path)

print(dataset.head())

nn_array = nearest_neighbor_tcrdist(dataset[:500], chain="both")