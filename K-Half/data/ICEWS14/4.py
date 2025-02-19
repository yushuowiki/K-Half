import pandas as pd

df = pd.read_csv('train.txt', delim_whitespace=True, header=None)

df.drop(columns=[4], inplace=True)

df.to_csv('train.txt', sep=' ', header=None, index=False)
