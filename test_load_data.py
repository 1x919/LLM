import pandas as pd

df = pd.read_csv("data/steam.csv")

print(df.shape)
print(df.columns)
print(df.head(3))
