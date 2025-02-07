import kalman_filter_acc_bound
import acceleration_vector
import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
datafile = os.path.join(script_dir, "data", "standing still.csv")  # Correctly join paths
if not os.path.exists(datafile):
    print(f"File not found: {datafile}")
df = pd.read_csv(datafile)  # Read the CSV file

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
datafile = os.path.join(script_dir, "data", "GroundyTruthy.csv")  # Correctly join paths
if not os.path.exists(datafile):
    print(f"File not found: {datafile}")
groundtruth = pd.read_csv(datafile)  # Read the CSV file

df = acceleration_vector.find_acceleration_magnitude(df)

pos_data = df['b3d']/100
acceleration_magnitudes = df['ta']/100
print(acceleration_magnitudes[0:10])

true_dist = groundtruth['b3d']

groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format='%H:%M:%S.%f')
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')

#we interpolate ground truth data to match shape of measurement data
true_times = groundtruth['timestamp']  # True timestamps
measurement_times = df['timestamp']  # Measurement timestamps
true_times_numeric = (true_times - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
measurement_times_numeric = (measurement_times - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
interpolated_positions = np.interp(measurement_times_numeric, true_times_numeric, true_dist)

if len(interpolated_positions) == len(pos_data):
    df["groundtruth"] = interpolated_positions
else:
    print('lengths dont match')

true_dist = df["groundtruth"]

#kalman_filter_pos_vel_acc.kalman_filter(pos_data, acceleration_data, df['timestamp'])
s, smooth_xs = kalman_filter_acc_bound.kalman_filter(pos_data, acceleration_magnitudes, df['timestamp'], smoothing=True)
#can use s, smooth_xs to update the dataframe. s is a saver object that stores a bunch of info about the filter.
#We will probably have to choose some things from s to store. 

xs = s.x
zs = s.z
cov = s.P

#plot_track(xs[:, 0], true_dist, zs, cov)

df['filtered'] = s.x[:, 0]

plt.figure()
plt.plot(df['timestamp'], true_dist, label="truth")
plt.plot(df['timestamp'], s.z, label="measurement")
plt.plot(df['timestamp'], s.x[:, 0], label="filter")
plt.plot(df['timestamp'], smooth_xs[:, 0], label="smooth") #might not use, cause movement is not super smooth
plt.title(f"R={s.R[0,0]}")
plt.legend()
plt.show()

'''import numpy as np 
plt.figure()
traces = [np.trace(cov) for cov in s.P]
plt.plot(traces)
plt.title("Trace of Covariance Over Time")
plt.show()'''
""" plt.figure()
plt.plot(groundtruth['timestamp'], true_dist, label="truth")
plt.legend() """




