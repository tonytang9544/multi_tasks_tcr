import pandas as pd

df = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    }
)

print(df[(df["a"] < 3) & (df["b"] > 4)])