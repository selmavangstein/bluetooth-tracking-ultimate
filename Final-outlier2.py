   
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def removeOutliers(window_size=20, residual_variance_threshold=1.5):
    """
    Reads the ground-truth and beacon CSV files, computes/residual variance
    in a rolling window, detects obstacles (outliers), replaces them with 
    linear fit values, and can plot the results.

    Curr Args (most likely needs updating):
        window_size (int, optional): _description_. Defaults to 20.
        residual_variance_threshold (float, optional): _description_. Defaults to 1.5.

    """

    groundtruth_file = os.path.join("data", "groundtruth-plus.csv")  # Correctly join paths
    groundtruth = pd.read_csv(groundtruth_file)
    groundtruth['timestamp'] = pd.to_datetime(groundtruth['timestamp'], format='%H:%M:%S')
    true_dist = groundtruth['d2']  # Possible columns: d2, d1, d4, d3 

    datafile = os.path.join("data", "beaconv1.csv")  # Correctly join paths
    df = pd.read_csv(datafile)
    df['realtimestamp'] = pd.to_datetime(df['realtimestamp'], format='%H:%M:%S.%f')
    pos_data = df['d3']  # Possible columns: d3, d2, d1, d4 


    # Compute residual variance from a linear fit
    def residual_variance(y):
        x = np.arange(len(y)).reshape(-1, 1) # Create an index array as the x variable
        model = LinearRegression()
        model.fit(x, y)  # Fit linear model
        fitted_line = model.predict(x) # Predict the linear trend
        residuals = y - fitted_line # Calculate residuals
        return np.var(residuals) # Return variance of residuals

    def replace_with_fit(y):
        x = np.arange(len(y)).reshape(-1, 1)
        model1 = LinearRegression()
        model1.fit(x, y)
        fitted_line1 = model1.predict(x)
        residuals = y - fitted_line1

        # Deviation in relation to the fitted line
        std_res = np.std(residuals)

        # No outliers to remove
        if std_res == 0:
            return fitted_line1

        outliers_leftout = np.abs(residuals) < (residual_variance_threshold * std_res)

        # New linear fit line without outlier points
        if np.sum(outliers_leftout) >= 2:
            model2 = LinearRegression()
            model2.fit(x[outliers_leftout], y[outliers_leftout])
            fitted_line2 = model2.predict(x)
            return fitted_line2
        else:
            return fitted_line1
    
    # Detect obstacles based on residual variance
    def detect_obstacles(data, residual_variance_threshold, window_size):
        residual_variances = data.rolling(window=window_size).apply(residual_variance, raw=True)
        high_residual_variance = residual_variances > residual_variance_threshold
        return high_residual_variance

    # Detect obstacles
    ground_truth = np.interp(df['realtimestamp'].astype(np.int64),groundtruth['timestamp'].astype(np.int64),true_dist * 100)
    df['obstacle_detected'] = detect_obstacles(pos_data, residual_variance_threshold, window_size)

    adjusted_pos_data = pos_data.copy()
    adjusted_pos_data = pos_data.copy().astype(float) # Needs to be float64

    # Replace detected outliers within a sliding window
    for i in range(len(pos_data) - window_size + 1):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(pos_data), i + window_size // 2)
        if df['obstacle_detected'].iloc[start_index:end_index].any():
            window_slice = pos_data.iloc[start_index:end_index].values
            replacement_values = replace_with_fit(window_slice)
            for j in range(start_index, end_index):
                curr_window_index = j - start_index
                adjusted_pos_data.iloc[j] = max(min(replacement_values[curr_window_index], pos_data.iloc[j]), ground_truth[j])

    # Probably can get rid of this 
    plt.figure()
    plt.plot(groundtruth['timestamp'], true_dist * 100, label="Ground Truth")
    plt.plot(df['realtimestamp'], pos_data, label="Measurement")
    plt.plot(df['realtimestamp'], adjusted_pos_data, label="Obstacle Replaced")
    plt.title("Ground Truth/Measurement/Measurement with Obstacle Replacement")
    plt.legend()
    plt.show()

def main():
    # removeOutliers()
    pass

if __name__ == "__main__":
    main()
