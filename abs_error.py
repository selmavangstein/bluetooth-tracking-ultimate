import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

"""This file calculates and plots the absolute error for each beacon over time.
Also plots the mean of the absolute errors of the four beacons over time.
Plots saved in "charts" folder."""

def interpolate_groundtruth(groundtruth, measurement_timestamps):
    """" Returns ground truth with interpolated data points aligned with measurement timestamps"""

    groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format="%H:%M:%S.%f")
    measurement_timestamps = pd.to_datetime(measurement_timestamps, format="%H:%M:%S.%f")

    groundtruth = groundtruth.set_index('timestamp')
    groundtruth = groundtruth.reindex(groundtruth.index.union(measurement_timestamps)).interpolate(method='time')
    groundtruth = groundtruth.loc[measurement_timestamps].reset_index()

    return groundtruth

def calculate_abs_error(groundtruth, measurements):
    """ Returns absolute error for each of the beacons at each measurement timestamp"""
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

def plot_abs_error(timestamps, errors, title="", plot=False):
    """ Creates a subplot for each beacon showing its absolute error at each measurement timestamp"""
    axes = plt.subplots(4,1, figsize=(10,12), sharex=True, sharey=True)[1]

    for i, ax in enumerate(axes,1):
        ax.plot(timestamps, errors[i], label=f'Absolute Error Beacon {i}', marker='o')
        ax.set_title(f'Absolute Error for Beacon {i} Over Time ({title})')
        ax.set_ylabel('Absolute Error (m)')
        ax.grid(True)
        ax.legend()
    plt.xlabel('Time')
    plt.savefig(os.path.join(os.getcwd(), f'charts/{title}-abserror.png'))
    if plot: plt.show()
    plt.close()

def plot_mean_abs_error(timestamps, mean_abs_error, title="", plot=False):
    """ Plots the mean of the absolute errors of the four beacons over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, mean_abs_error, label='Mean Absolute Error Across Beacons', color='green', marker='x')
    plt.title(f'Mean Absolute Error Across All Beacons Over Time ({title})')
    plt.xlabel('Time')
    plt.ylabel('Mean Absolute Error (m)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), f'charts/{title}-meanabserror.png'))
    if plot: plt.show()
    plt.close()
