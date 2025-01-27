import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

def calculate_abs_error(groundtruth, measurements):

    groundtruth[['d1','d2','d3','d4']] *= 100

    groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'])
    measurements['timestamp'] = pd.to_datetime(measurements['timestamp']).dt.floor('S')

    merged = pd.merge(groundtruth, measurements, on='timestamp')

    merged['abs_error'] = np.abs(merged['b3d'] - merged['d2'])
    merged['abs_error_percentage'] = (merged['abs_error'] / merged['d2']) * 100

    return merged

def plot_abs_error(timestamps, abs_error):

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, abs_error, label='Absolute Error', color='blue', marker='o')
    plt.title('Absolute Error Over Time')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error (cm)')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_abs_error_percentage(timestamps, abs_error_percentage):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, abs_error_percentage, label='Absolute Error Percentage', color='green', marker='x')
    plt.title('Absolute Error Percentage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error Percentage')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))  
    datafile = os.path.join(script_dir, "data", "4beaconv1.csv")
    if not os.path.exists(datafile):
        print(f"File not found: {datafile}")
    measurements = pd.read_csv(datafile)

    script_dir = os.path.dirname(os.path.abspath(__file__))  
    datafile = os.path.join(script_dir, "data", "groundtruth-plus.csv")  
    if not os.path.exists(datafile):
        print(f"File not found: {datafile}")
    groundtruth = pd.read_csv(datafile)

    filtered_data = calculate_abs_error(groundtruth, measurements)
    print(filtered_data)

    plot_abs_error(filtered_data['timestamp'], filtered_data['abs_error'])
    plot_abs_error_percentage(filtered_data['timestamp'], filtered_data['abs_error_percentage'])