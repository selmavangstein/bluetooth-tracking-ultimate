''' variance analysis,
and replaces obstacle data with data on the linear fit to the window'''
'''Removed symmetry'''

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from smoothing import *
from trilateration import trilateration_two

window = 10
window_size = window
symmetry_threshold = 0.5
residual_variance_threshold = 1.5

beacon1 = (0, 0)
beacon2 = (30, 0)
beacon3 = (15, 26)

# filename="first_test"
# cleaned_data_file = clean_data(filename+'.log')[0]
# df_data = pd.read_csv(cleaned_data_file)
# df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

# with new data format
filename = "first_test.log"
cleaned_file = clean_new_format_data(filename)
df_data = pd.read_csv(cleaned_file)
df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])


xs = []
ys = []
# add a beacon position column (x,y) to the csv
for i in range(len(df_data)):
    # get distance from beacons
    si = df_data['beacon1_dist'][i]
    sj = df_data['beacon2_dist'][i]
    sk = df_data['beacon3_dist'][i]

    pos = trilateration_two(beacon1, si, beacon2, sj, beacon3, sk)
    x = pos[0]
    y = pos[1]

    xs.append(x)
    ys.append(y)


print(xs, ys)
# add xs and ys to df
df_data['x'] = pd.DataFrame(xs)
df_data['y'] = pd.DataFrame(ys)

# save to csv
df_data.to_csv(filename + "_cleaned.csv", index=False)

print(df_data)