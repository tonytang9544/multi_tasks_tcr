import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# a = {
#     "3": [0, 1],
#     "5": [23, 3]
# }

# b = pd.DataFrame(a)
# cols = np.array([int(i) for i in b.columns], dtype=np.int16)
# c = b.to_numpy()
# corr = c[0] / c.dot((1, 1))

# print(corr)
# plt.scatter(x=cols, y=corr)
# plt.savefig("1.png")
# plt.cla()
# plt.close()

a = [
    "abc",
    "def"
]

with open("test.txt", "w") as f:
    f.write("\n".join(a))