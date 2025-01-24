import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import zscore

# Ground truth data
groundtruth_file = os.path.join("data", "groundtruth-plus.csv")  # Correctly join paths
groundtruth = pd.read_csv(groundtruth_file)
groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format='%H:%M:%S')
true_dist = groundtruth['d3']

# Measurement data
datafile = os.path.join("data", "beaconv1.csv")  # Correctly join paths
df = pd.read_csv(datafile)
df['realtimestamp'] = pd.to_datetime(df['realtimestamp'], format='%H:%M:%S.%f')
pos_data = df['d4'] 

# Constraints/thresholds
window_size = 10
zscore_threshold = 2.5

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

# Define the detect_outliers function using Z-score
def detect_outliers(data, zscore_threshold):
    z_scores = zscore(data)  # Compute z-scores of the data
    return ((np.abs(z_scores) > zscore_threshold) | (z_scores < 0))

# Detect outliers
ground_truth = np.interp(df['realtimestamp'].astype(np.int64), groundtruth['timestamp'].astype(np.int64), true_dist * 100)
outliers = detect_outliers(pos_data, zscore_threshold)

adjusted_pos_data = pos_data.copy()
adjusted_pos_data = adjusted_pos_data.astype(float) # Needs to be float64

# Replace detected outliers with a fitted value
for i in range(len(pos_data) - window_size + 1):
    start_index = max(0, i - window_size // 2)
    end_index = min(len(outliers), i + window_size // 2)
    window_slice = outliers[start_index:end_index]
    if any(window_slice):
        replacement_values = replace_with_fit(pos_data.iloc[i: i + window_size].values)
        # Make sure values stay above the ground truth line
        for j in range(i, i + window_size):
            adjusted_pos_data.iloc[j] = max(replacement_values[j - i], ground_truth[j])
    else:
        # Make sure line stays above the ground truth line
        for j in range(i, i + window_size):
            if adjusted_pos_data.iloc[j] < ground_truth[j]:
                adjusted_pos_data.iloc[j] = ground_truth[j]


plt.figure()
plt.plot(groundtruth['timestamp'], true_dist * 100, label="Ground Truth")
plt.plot(df['realtimestamp'], pos_data, label="Measurement")
plt.plot(df['realtimestamp'], adjusted_pos_data, label="Outlier Replaced")
plt.title("Ground Truth/Measurement/Outlier Replaced Lines")
plt.legend()
plt.show()