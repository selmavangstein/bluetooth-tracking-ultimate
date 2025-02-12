import pandas as pd
from datacleanup import cleanup_file

filename = "data/feb9/2-9-testB-ftm.csv"
df = pd.read_csv(filename)
df = cleanup_file(df)

columns_to_divide = ["b1d", "b2d", "b3d", "b4d", "xa", "ya", "za"]
df[columns_to_divide] = df[columns_to_divide].apply(pd.to_numeric, errors='coerce')

# Divide only the selected columns by 100
df[columns_to_divide] = df[columns_to_divide] / 100

# Overwrite the original CSV file
df.to_csv(filename, index=False)