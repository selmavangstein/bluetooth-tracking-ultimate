
"""
Functions for removing outliers from a dataframe.
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def removeOutliers(df, window_size=10, residual_variance_threshold=1.5):
    """Removes outliers from a dataframe using a rolling residual variance threshold.

    Args:
        df (pandas df): _description_
        window_size (int, optional): _description_. Defaults to 10.
        residual_variance_threshold (float, optional): _description_. Defaults to 1.5.

    Returns:
        pandas df: df of the same format as the input with all the outliers removed
    """

    df = df.copy()

    def residual_variance(y):
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        fitted_line = model.predict(x)
        residuals = y - fitted_line
        return np.var(residuals)

    # Ask Selma if it is ok to only return the first value of the fitted line
    def replace_with_fit(y):
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        fitted_line = model.predict(x)
        return fitted_line

    # Do the outlier removal for each distance column
    for column in df.columns:
        if not column.startswith('b'):
            continue
                   
        df[column] = df[column].ffill()

        df[f'{column}_residual_variance'] = df[column].rolling(window=window_size).apply(residual_variance, raw=True)

        df[f'{column}_outlier_detected'] = (
            df[f'{column}_residual_variance'] > residual_variance_threshold
        )

        for i in range(len(df) - window_size + 1):
            if df[f'{column}_outlier_detected'].iloc[i - window_size//2 :i + window_size//2].any():
                df.loc[df.index[i:i + window_size], column] = replace_with_fit(df[column].iloc[i:i + window_size].values)

        df[f'{column}_outlier_free'] = ~df[f'{column}_outlier_detected']

    # Remove all columns after a specific column (in this case column za)
    # Even though we are removing outlier data here we are keeping all the relevant columns needed for the rest of post-processing
    # I assume this column will change after we add the compass data
    df = df.loc[:, :'za']

    return df

def outliersRemoved(df, window_time='800ms', residual_variance_threshold=0.5):
    """Removes outliers from a dataframe using a rolling residual variance threshold and linear regression.

    Args:
        df (pandas df): _description_
        window_time (str): _description_. Defaults to 800ms.
        residual_variance_threshold (float, optional): _description_. Defaults to 0.5.

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

    # Function that replaces with fit for the OG window if not enough inlier points are in our adjusted window
    def replace_with_og(y):
         if len(y) < 2:
             return y
         x = np.arange(len(y)).reshape(-1, 1)
 
         nan_i = []
         for i in range(len(y)): 
             if np.isnan(y[i]):
                 nan_i.append(i)
                 y[i] = (len(y) // 2)
 
         model1 = LinearRegression()
         model1.fit(x,y)
         fitted_line1 = model1.predict(x)
 
         no_outliers = np.abs(y - np.median(y)) < 0.5
 
         for i in range(len(y)): 
             if i in nan_i:
                 y[i] = np.nan
 
         # New linear fit line without outlier points
         if np.sum(no_outliers) >= 2:
             model2 = LinearRegression()
             model2.fit(x[no_outliers], y[no_outliers])
             fitted_line2 = model2.predict(x) 
             return fitted_line2
         else:
             return fitted_line1
    # Replaces outliers in the OG window, considering both +/- spikes around 0.5. 
    def replace_with_fit1(y):
         if len(y) < 2:
             return y
         x = np.arange(len(y)).reshape(-1, 1)
         model1 = LinearRegression()
         model1.fit(x, y)
         fitted_line1 = model1.predict(x)
         residuals = y - fitted_line1 
 
         no_outliers = np.abs(y - np.median(y)) < 0.5
 
         # New linear fit line without outlier points
         if np.sum(no_outliers) >= 2:
             model2 = LinearRegression()
             model2.fit(x[no_outliers], y[no_outliers])
             fitted_line2 = model2.predict(x) 
             return fitted_line2
         else:
             return replace_with_og(window)
 # Replaces outliers in the adjusted window, only considering large spikes over 0.5
    def replace_with_fit2(y, window):
        if len(y) < 2:
            return y
        x = np.arange(len(y)).reshape(-1, 1)
        model1 = LinearRegression()
        model1.fit(x,y)
        fitted_line1 = model1.predict(x)

        # Compares the difference between each data point minus the median of the window against the threshold
        # and gives it a boolean value not taking np.abs
        no_outliers = (y - np.median(y)) < 0.5

        # Replace vlaues that are outliers with Nan
        false_index = np.where(no_outliers == False)[0] 
        false = 0
        if false_index.any():
            for val in false_index:
                false = val
                y[false] = np.nan

        # New linear fit line without outlier points
        if np.sum(no_outliers) >= 2:
            model2 = LinearRegression()
            model2.fit(x[no_outliers], y[no_outliers])
            fitted_line2 = model2.predict(x) 
            return fitted_line2
        else:
            return fitted_line1

    # Detect obstacles based on residual variance and creates a boolean col in our df whether it is an outlier or not
    def detect_obstacles(data, residual_variance_threshold, window_time):
         residual_variances = data.rolling(window_time).apply(residual_variance, raw=True)
         residual_variances.dropna(inplace=True)
         obstacles = residual_variances > residual_variance_threshold
         return obstacles
 
     # Do the outlier removal for each distance column
    for column in df.columns:
        if not column.startswith('b'):
            continue

        # Dealing with NaN values
        df[column] = df[column].ffill()

        df[f'{column}_obstacle_detected'] = detect_obstacles(df[column], residual_variance_threshold, window_time)
        adjusted_col_data = df[column].copy().astype(float)

        for current_time in df.index:
            start_time = current_time - (pd.Timedelta(window_time) / 2)
            end_time   = current_time + (pd.Timedelta(window_time) / 2)

            if df[f'{column}_obstacle_detected'].loc[start_time:end_time].any():
                window = adjusted_col_data.loc[start_time:end_time].values
                replacement_vals = replace_with_fit2(replace_with_fit1(window), window) 
                adjusted_index = adjusted_col_data.loc[start_time:end_time].index
                adjusted_col_data.loc[adjusted_index] = replacement_vals

        df[column] = adjusted_col_data
 
 
     # Timestamps are indexed, need to be back in forms of cols
    df.reset_index(inplace=True)
    df = df.loc[:, :'za']
 
    df.to_csv(f"processedRemoveOutliers.csv", index=False)
    return df

def pipelineRemoveOutliers(df, window_size=20, residual_variance_threshold=0.8):
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

        df[column] = df[column].ffill() # forward fill those NaN values 
        
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



