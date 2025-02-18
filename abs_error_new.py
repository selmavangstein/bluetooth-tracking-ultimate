import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

def interpolate_groundtruth(groundtruth, measurement_timestamps):

    groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format="%H:%M:%S.%f")
    measurement_timestamps = pd.to_datetime(measurement_timestamps, format="%H:%M:%S.%f")

    groundtruth = groundtruth.set_index('timestamp')
    groundtruth = groundtruth.reindex(groundtruth.index.union(measurement_timestamps)).interpolate(method='time')
    groundtruth = groundtruth.loc[measurement_timestamps].reset_index()

    return groundtruth

def calculate_abs_error(groundtruth, measurements):

    measurements['timestamp'] = pd.to_datetime(measurements['timestamp'], format="%H:%M:%S.%f")

    groundtruth = interpolate_groundtruth(groundtruth, measurements['timestamp'])
    groundtruth_rename = groundtruth.rename(columns={
        'b1d': 'b1d_gt', 'b2d': 'b2d_gt', 'b3d': 'b3d_gt', 'b4d': 'b4d_gt'
    })

    merged = pd.merge(groundtruth_rename, measurements, on='timestamp')

    errors = {}
    for i in range(1,5):
        merged[f'abs_error_b{i}'] = np.abs(merged[f'b{i}d_gt'] - merged[f'b{i}d'])
        errors[i] = merged[f'abs_error_b{i}']

    merged['mean_abs_error'] = merged[[f'abs_error_b{i}' for i in range (1,5)]].mean(axis=1)

    return merged, errors

def plot_abs_error(timestamps, errors):
    axes = plt.subplots(4,1, figsize=(10,12), sharex=True)[1]

    for i, ax in enumerate(axes,1):
        ax.plot(timestamps, errors[i], label=f'Absolute Error Beacon {i}', marker='o')
        ax.set_title(f'Absolute Error Over Time (Beacon {i})')
        ax.set_ylabel('Absolute Error (m)')
        ax.grid(True)
        ax.legend()
    plt.xlabel('Time')
    plt.savefig(os.path.join(os.getcwd(), f'charts/abserror.png'))
    plt.show()
    plt.close()

def plot_mean_abs_error(timestamps, mean_abs_error):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, mean_abs_error, label='Mean Absolute Error Across Beacons', color='green', marker='x')
    plt.title('Mean Absolute Error Across All Beacons Over Time')
    plt.xlabel('Time')
    plt.ylabel('Mean Absolute Error (m)')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.close()
