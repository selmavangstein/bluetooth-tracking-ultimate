import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter_pos_vel_acc import pipelineKalman

"""A selection of functions to test for data characteristics.
Is useful in tuning the Kalman filter.
"""

def testing_variance(df):
    print("TESTING VARIANCE")
    # Compute variance of the 'distance' column
    variance1 = df['b1d'].var()
    print("Variance 1:", variance1)

    variance2 = df['b2d'].var()
    print("Variance 2:", variance2)

    variance3 = df['b3d'].var()
    print("Variance 3:", variance3)

    variance4 = df['b4d'].var()
    print("Variance 4:", variance4)

    print("ave var: ", (variance1+variance2+variance3)/3) #ignoring 4, had an obstacle


def time_fluctuation(df):
    #TIME FLUCTUATION CHECK
    print("dt FLUCTUATION CHECK")
    # Load CSV file
    df = pd.read_csv("data/standing still.csv", parse_dates=['timestamp'])  # Ensure timestamps are datetime

    # Compute time differences in seconds
    time_diffs = df['timestamp'].diff().dt.total_seconds()

    # Compute the average time difference
    avg_diff = time_diffs.mean()

    # Count how many rows exceed 10% fluctuation
    num_exceeded = (time_diffs > (1.05 * avg_diff)).sum()

    # Print the result
    print(f"Number of rows with time differences exceeding 10%: {num_exceeded}")

def systematic_error(df):
    print("SYSTEM ERROR CHECK")
    mean1 = df['b1d'].mean()
    print("Error 1:", mean1-10.8)

    mean2 = df['b2d'].mean()
    print("Error 2:", mean2-10.8)

    mean3 = df['b3d'].mean()
    print("Error 3:", mean3-10.8)

def residual_stats(df):
    print("COMPUTING STATISTICS ON RESIDUALS")
    processed_df, savers = pipelineKalman(df)
    for saver in savers:
        stationary_residuals = saver.y

        # Compute statistics:
        sigma = np.std(stationary_residuals)
        variance = np.var(stationary_residuals)
        print("Standard Deviation (sigma):", sigma)
        print("Variance:", variance)

        # Choose a multiplier (e.g., k = 3 for 3-sigma):
        k = 3
        variance_threshold = (k * sigma) ** 2
        print("Initial variance_threshold (using k = {}): {:.4f}".format(k, variance_threshold))

        # Plot histogram of residuals to visualize:
        plt.hist(stationary_residuals, bins=30, alpha=0.7, label='Residuals')
        plt.axvline(k*sigma, color='red', linestyle='--', label=f'+{k}σ')
        plt.axvline(-k*sigma, color='red', linestyle='--', label=f'-{k}σ')
        plt.title("Histogram of Stationary Residuals")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()


def average_absolute_error(df, gt):
    diff1 = (df['b1d']-gt['b1d'])
    diff2 = (df['b2d']-gt['b2d'])
    diff3 = (df['b3d']-gt['b3d'])
    diff4 = (df['b4d']-gt['b4d'])

    print("Error1: ", diff1.mean())
    print("Error2: ", diff2.mean())
    print("Error3: ", diff3.mean())
    print("Error4: ", diff4.mean())

df = pd.read_csv("data/OverTheHeadTest.csv")
gt = pd.read_csv("data/GT-overheadtest-UWB-feb5.csv")

df = pd.read_csv("data/standing still.csv")
#average_absolute_error(df,gt)
systematic_error(df)