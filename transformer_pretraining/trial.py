import numpy as np

change_proportion = 0.15
rand_num = np.random.rand(3, 2)
change_mask = rand_num <= change_proportion
print(change_mask)