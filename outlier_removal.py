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

# print(df_data)
exit()


# Define a function to compute residual variance from a linear fit
def residual_variance(y):
    x = np.arange(len(y)).reshape(-1, 1)  # Create an index array as the x variable
    model = LinearRegression()
    model.fit(x, y)  # Fit linear model
    fitted_line = model.predict(x)  # Predict the linear trend
    residuals = y - fitted_line  # Calculate residuals
    return np.var(residuals)  # Return variance of residuals

# Define a function to replace obstacle points with values from the linear fit
def replace_with_fit(y):
    x = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    fitted_line = model.predict(x)
    return fitted_line

# Use your custom smoothing function for RSSI and distance
windows = [window]
smas_rssi, timestamps = plot_smoothed_data(filename + "_cleaned.csv", windows, param='rssi')
smas_distance, _ = plot_smoothed_data(filename + "_cleaned.csv", windows, param='distance')


# only things we need
df_data['rssi_smooth'] = smas_rssi[0]
df_data['distance_smooth'] = smas_distance[0]

# Drop rows with NaN values due to smoothing at the edges
df_data.dropna(subset=['rssi_smooth', 'distance_smooth'], inplace=True)

''' # Normalize smoothed values using z-score
df_data['rssi_norm'] = zscore(df_data['rssi_smooth'])
df_data['distance_norm'] = zscore(df_data['distance_smooth'])

# Compute the rolling correlation as a measure of symmetry
window_size = window
df_data['symmetry_score'] = -df_data['rssi_norm'].rolling(window=window_size, center=True).corr(df_data['distance_norm']) '''

# Calculate the rolling residual variance for rssi_smooth and distance_smooth
df_data['rssi_residual_variance'] = df_data['rssi'].rolling(window=window_size).apply(residual_variance, raw=True)
df_data['distance_residual_variance'] = df_data['distance'].rolling(window=window_size).apply(residual_variance, raw=True)

''' # Detect obstacles based on either symmetry or residual variance conditions
df_data['obstacle_detected'] = (df_data['symmetry_score'] > symmetry_threshold) & (
    (df_data['rssi_residual_variance'] > residual_variance_threshold) |
    (df_data['distance_residual_variance'] > residual_variance_threshold)
) '''
# Detect obstacles based on residual variance conditions
df_data['obstacle_detected'] = (
    (df_data['rssi_residual_variance'] > residual_variance_threshold) |
    (df_data['distance_residual_variance'] > residual_variance_threshold)
)

# Replace obstacle data points with fitted line values in the specified window
for i in range(len(df_data) - window_size + 1):
    if df_data['obstacle_detected'].iloc[i - window_size//2 :i + window_size//2].any():
        df_data.loc[df_data.index[i:i + window_size], 'rssi_smooth'] = replace_with_fit(df_data['rssi_smooth'].iloc[i:i + window_size].values)
        df_data.loc[df_data.index[i:i + window_size], 'distance_smooth'] = replace_with_fit(df_data['distance_smooth'].iloc[i:i + window_size].values)

# Check your adjusted data
#print(df_data[['timestamp', 'rssi_smooth', 'distance_smooth', 'obstacle_detected']])

df_data['obstacle_free'] = ~df_data['obstacle_detected']

# Create DataFrames with obstacle and obstacle-free data points
obstacle_data = df_data[df_data['obstacle_detected']]
obstacle_free_data = df_data[df_data['obstacle_free']]

print(len(df_data['distance']))
print(len(obstacle_data['distance']))
print(len(obstacle_free_data['distance']))

