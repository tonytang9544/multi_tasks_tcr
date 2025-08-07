import torch
import pandas as pd

loss = torch.nn.CrossEntropyLoss()

a = torch.rand(3, 5)
b = torch.tensor([1,2,4])

print(loss(a, b))

print(" ".join("abc"))

mask = torch.rand(3) > 0.5
print(mask)

df = pd.Series(
    [1, 2, 3]
)


df2 = pd.Series(
    [10, 20, 30]
)

print(df)
print(df.where(mask, df2))

batch_size = 5
random_mask = torch.rand(batch_size) > 0.5
labels = torch.ones(batch_size, dtype=torch.int8)
labels[random_mask] = 0
print(labels)