import numpy as np
import pandas as pd
import os

""" # Load your CSV file
file_path = 'bluetooth-tracking-ultimate/data/groundtruth.csv'
groundtruth_data = pd.read_csv(file_path)

# Ensure the timestamp column is in datetime format
groundtruth_data['timestamp'] = pd.to_datetime(groundtruth_data['timestamp'], format='%H:%M:%S')
additional_rows = groundtruth_data.copy()
additional_rows['timestamp'] += pd.Timedelta(seconds=1)
combined_data = pd.concat([groundtruth_data, additional_rows]).sort_values(by='timestamp')
combined_data['timestamp'] = combined_data['timestamp'].dt.strftime('%H:%M:%S')
output_path = 'bluetooth-tracking-ultimate/data/groundtruth-plus.csv'
combined_data.to_csv(output_path, index=False) """

# Read the CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
datafile = os.path.join(script_dir, "data", "beaconv1.csv")  # Correctly join paths
if not os.path.exists(datafile):
    print(f"File not found: {datafile}")
df = pd.read_csv(datafile)  # Read the CSV file

datafile = os.path.join(script_dir, "data", "groundtruth-plus.csv")  # Correctly join paths
if not os.path.exists(datafile):
    print(f"File not found: {datafile}")
groundtruth = pd.read_csv(datafile)  # Read the CSV file

# Convert timestamps to datetime
groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format='%H:%M:%S')
df['realtimestamp'] = pd.to_datetime(df['realtimestamp'], format='%H:%M:%S.%f')

# Extract the true positions and measurements
true_positions = np.array(groundtruth['d1'])  # Your array of true positions
true_times = groundtruth['timestamp']  # True timestamps
measurements = np.array(df['d1'])   # Your array of measurements
measurement_times = df['realtimestamp']  # Measurement timestamps

# Convert datetime to numeric (seconds since epoch)
true_times_numeric = (true_times - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
measurement_times_numeric = (measurement_times - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

# Interpolate true positions to match the measurement times
interpolated_positions = np.interp(measurement_times_numeric, true_times_numeric, true_positions)

if len(interpolated_positions) == len(measurements):
    df["groundtruth"] = interpolated_positions
else:
    print('lengths dont match')

# Now calculate residuals (measurements - interpolated true positions)
residuals = measurements - interpolated_positions

# Compute variance of residuals
variance = np.var(residuals)

print(f"Variance of residuals: {variance}")

