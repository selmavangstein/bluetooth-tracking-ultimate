import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

def removeOutliers_dp(df, window_size=20, residual_variance_threshold=0.8):
    """Removes outliers from a dataframe using a rolling residual variance threshold and standard deviation.

    Args:
        df (pandas df): _description_
        window_size (int, optional): _description_. Defaults to 20.
        residual_variance_threshold (float, optional): _description_. Defaults to 0.8.

    Returns:
        pandas df: df of the same format as the input with all the outliers removed
    """
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')

    # Compute residual variance from a linear fit
    def residual_variance(y):
        if len(y) < 2:
            return 0
        x = np.arange(len(y)).reshape(-1, 1) # Create an index array as the x variable
        model = LinearRegression()
        model.fit(x, y)  # Fit linear model
        fitted_line = model.predict(x) # Predict the linear trend
        residuals = y - fitted_line # Calculate residuals
        return np.var(residuals) # Return variance of residuals

    def replace_with_fit(y):
        if len(y) < 2:
            return y
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

        # Compares the residuals with the threshold and sets it to either true or false for each data point in the window
        outliers_notleftout = np.abs(residuals) < (residual_variance_threshold *std_res)
        
        # New linear fit line without outlier points
        if np.sum(outliers_notleftout) >= 2:
            model2 = LinearRegression()
            # Excludes points that are "outliers"
            model2.fit(x[outliers_notleftout], y[outliers_notleftout])
            fitted_line2 = model2.predict(x)
            return fitted_line2
        else:
            return fitted_line1

    # Detect obstacles based on residual variance
    def detect_obstacles(data, residual_variance_threshold, window_size):
        residual_variances = data.rolling(window=window_size).apply(residual_variance, raw=True)
        obstacles = residual_variances > residual_variance_threshold
        return obstacles

    # Do the outlier removal for each distance column
    for column in df.columns:
        if not column.startswith('b'):
            continue
        
        df[f'{column}_obstacle_detected'] = detect_obstacles(df[column], residual_variance_threshold, window_size)
        adjusted_col_data = df[column].copy().astype(float)

        for i in range(len(df) - window_size + 1):
            start_index = max(0, i - window_size // 2)
            end_index = min(len(df), i + window_size // 2)

            if df[f'{column}_obstacle_detected'].iloc[start_index:end_index].any():
                window = df[column].iloc[start_index:end_index].values
                replacement_vals = replace_with_fit(window)
                for j in range(start_index, end_index):
                    curr_window_index = j - start_index
                    adjusted_col_data.iloc[j] = replacement_vals[curr_window_index]

        df[f'{column}_adjusted'] = adjusted_col_data
    
    return df

def removeOutliers_ts(df, window_time='2000ms', residual_variance_threshold=0.8):
    """Removes outliers from a dataframe using a rolling residual variance threshold and standard deviation.

    Args:
        df (pandas df): _description_
        window_time (str): _description_. Defaults to 2000ms (10 timestamps per second, however may need to be changed).
        residual_variance_threshold (float, optional): _description_. Defaults to 0.8.

    Returns:
        pandas df: df of the same format as the input with all the outliers removed
    """

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')

    # Sort timestamps even though they should naturally be sorted and index them
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)

    # Compute residual variance from a linear fit
    def residual_variance(y):
        if len(y) < 2:
            return 0
        x = np.arange(len(y)).reshape(-1, 1) # Create an index array as the x variable
        model = LinearRegression()
        model.fit(x, y)  # Fit linear model
        fitted_line = model.predict(x) # Predict the linear trend
        residuals = y - fitted_line # Calculate residuals
        return np.var(residuals) # Return variance of residuals

    def replace_with_fit(y):
        if len(y) < 2:
            return y
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

        # Compares the residuals with the threshold and sets it to either true or false for each data point in the window
        outliers_notleftout = np.abs(residuals) < (residual_variance_threshold *std_res)
        
        # New linear fit line without outlier points
        if np.sum(outliers_notleftout) >= 2:
            model2 = LinearRegression()
            # Excludes points that are "outliers"
            model2.fit(x[outliers_notleftout], y[outliers_notleftout])
            fitted_line2 = model2.predict(x)
            return fitted_line2
        else:
            return fitted_line1

    # Detect obstacles based on residual variance
    def detect_obstacles(data, residual_variance_threshold, window_time):
        residual_variances = data.rolling(window_time).apply(residual_variance, raw=True)
        obstacles = residual_variances > residual_variance_threshold
        return obstacles

    # Do the outlier removal for each distance column
    for column in df.columns:
        if not column.startswith('b'):
            continue

        df[f'{column}_obstacle_detected'] = detect_obstacles(df[column], residual_variance_threshold, time_window)
        adjusted_col_data = df[column].copy().astype(float)

        for current_time in df.index:
            start_time = current_time - (pd.Timedelta(window_time) / 2)
            end_time   = current_time + (pd.Timedelta(window_time) / 2)

            if df[f'{column}_obstacle_detected'].loc[start_time:end_time].any():
                window = adjusted_col_data.loc[start_time:end_time].values
                replacement_vals = replace_with_fit(window)
                adjusted_index = adjusted_col_data.loc[start_time:end_time].index
                adjusted_col_data.loc[adjusted_index] = replacement_vals

        df[f'{column}_adjusted'] = adjusted_col_data

    # Timestamps are indexed, need to be back in forms of cols
    df.reset_index(inplace=True)
    return df

def main():
    # removeOutliers_dp()
    # removeOutliers_ts()
    pass

if __name__ == "__main__":
    main()
