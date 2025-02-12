import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

def interpolate_groundtruth(groundtruth, measurement_timestamps):

    groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format="%H:%M:%S")
    measurement_timestamps = pd.to_datetime(measurement_timestamps, format="%H:%M:%S.%f")

    groundtruth = groundtruth.set_index('timestamp')
    groundtruth = groundtruth.reindex(groundtruth.index.union(measurement_timestamps)).interpolate(method='time')
    groundtruth = groundtruth.loc[measurement_timestamps].reset_index()

    return groundtruth

def calculate_abs_error(groundtruth, measurements):

    #measurements[['b1d','b2d','b3d','b4d']] /= 100

    measurements['timestamp'] = pd.to_datetime(measurements['timestamp'], format="%H:%M:%S.%f")

    groundtruth = interpolate_groundtruth(groundtruth, measurements['timestamp'])
    merged = pd.merge(groundtruth, measurements, on='timestamp')

    merged['abs_error'] = np.abs(merged['b3d'] - merged['d3'])
    merged['abs_error_percentage'] = (merged['abs_error'] / merged['d3']) * 100

    mean_error = merged['abs_error'].mean()
    print(f"Mean Absolute Error: {mean_error:.2f} m")

    return merged, mean_error

def plot_abs_error(timestamps, abs_error, mean_error, plot=False, title=""):

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, abs_error, label='Absolute Error', color='blue', marker='o')
    plt.title(f'Absolute Error Over Time - {title}')
    plt.xlabel('Time')
    plt.ylabel('Absolute Error (m)')
    plt.grid(True)
    plt.legend()

    text_box = f"Mean Absolute Error: {mean_error: .2f} m"
    plt.figtext(0.15, 0.85, text_box, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(os.getcwd(), f'charts/{title}-abserror.png'))
    if plot: plt.show()
    plt.close()

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
 
    datafile = os.path.join(script_dir, "data", "jan17-groundtruth.csv")  
    if not os.path.exists(datafile):
        print(f"File not found: {datafile}")
    groundtruth = pd.read_csv(datafile)

    filtered_data, mean_error = calculate_abs_error(groundtruth, measurements)
    # print(filtered_data)

    plot_abs_error(filtered_data['timestamp'], filtered_data['abs_error'], mean_error)
    # plot_abs_error_percentage(filtered_data['timestamp'], filtered_data['abs_error_percentage'])