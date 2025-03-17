import pandas as pd
import numpy as np
df = pd.read_csv("processedtest6.csv")

for index, row in df.iterrows():
    if index == 0: continue # skip first row
    distances = np.array([row[col] for col in df.columns if col.startswith('b')])
    if index == 1:
        print(distances)
        break