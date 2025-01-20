import kalman_filter_pos_vel_acc
import kalman_filter_pos_vel
import acceleration_vector
import pandas as pd
import os
from matplotlib import pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
datafile = os.path.join(script_dir, "data", "beaconv1.csv")  # Correctly join paths
if not os.path.exists(datafile):
    print(f"File not found: {datafile}")
df = pd.read_csv(datafile)  # Read the CSV file

script_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
datafile = os.path.join(script_dir, "data", "groundtruth-plus.csv")  # Correctly join paths
if not os.path.exists(datafile):
    print(f"File not found: {datafile}")
groundtruth = pd.read_csv(datafile)  # Read the CSV file

pos_data = df['d1']
acceleration_data = df['ax']

true_dist = groundtruth['d4']

groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format='%H:%M:%S')
df['realtimestamp'] = pd.to_datetime(df['realtimestamp'], format='%H:%M:%S.%f')

#kalman_filter_pos_vel_acc.kalman_filter(pos_data, acceleration_data, df['realtimestamp'])
s, smooth_xs = kalman_filter_pos_vel.kalman_filter(pos_data, df['realtimestamp'])
plt.figure()
plt.plot(groundtruth['timestamp'], true_dist*100, label="truth")
plt.plot(df['realtimestamp'], s.z, label="measurement")
plt.plot(df['realtimestamp'], s.x[:, 0], label="filter")
plt.plot(df['realtimestamp'], smooth_xs[:, 0], label="smooth") #might not use, cause movement is not super smooth
plt.title(f"R={s.R[0,0]}")
plt.legend()

""" plt.figure()
plt.plot(groundtruth['timestamp'], true_dist, label="truth")
plt.legend() """

plt.show()


