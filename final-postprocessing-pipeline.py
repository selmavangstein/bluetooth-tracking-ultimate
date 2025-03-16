
"""
Inputs one csv from each player or the plain txt file from each player (and then turns that into a csv)
Outputs an animated chart of the players' movements and 1d charts of the players' distances from each beacon

The program will only process data in columns that starts with a 'b' (for beacon data) (Ex. b1d) 
Lets keep that naming convention for the beacon data columns so we can add as many as we like without changing the code

Uses a pandas df to store the data, and a matplotlib animation to animate the data
"""

from GroundTruthPipeline import GroundTruthPipeline
from abs_error import *
from kalman_filter_pos_vel_acc import pipelineKalman
from kalman_2d import pipelineKalman_2d
#from kalman2d_4state import pipelineKalman_2d
from final_trilateration import trilaterate
from test_trilateration_v2 import weighted_trilateration
from report import *
#from kalman_filter_acc_bound import pipelineKalman

import joblib
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
import numpy as np
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.kalman import KalmanFilter
import os
import matplotlib.pyplot as plt
from datacleanup import cleanup_file

def loadData(filename):
    """
    Loads the data from a csv file into a pandas dataframe
    """
    return pd.read_csv(filename)

def smoothData(df, window_size=5):
    """
    Smooths the data in the dataframe using Exponential Moving Average (EMA)
    """
    smoothed_df = df.copy()
    for column in df.columns:
        # Only smooth columns that are beacon data
        if not column.startswith('b'):
            continue
        smoothed_df[column] = df[column].ewm(span=window_size, adjust=False).mean()
    return smoothed_df


def twoD_correction(locations, timestamps, acc, ema_window=100):
    """
    Attempts to correct the 2d trilateration data if there are big jumps
    Almost like another kalman filter (but 2d)

    takes in a list like : [[x,y], [x1,y1], ... [xn,yn]]
    """

    # Store the corrected data
    corrections = [locations[0]]

    # calculate an ema of the data
    # convert to a pandas dataframe
    locations_df = pd.DataFrame(locations, columns=['x', 'y'])

    # comvert timestamps to numbers
    timestamps = [pd.Timestamp(ts).timestamp() for ts in timestamps]

    corrected = 0
    total = 0

    # Loop through the data
    for i in range(1, len(locations)):
        # if the distance between two points is greater than 10m
        # get the time difference between the points
        time_diff = (timestamps[i] - timestamps[i-1])
        
        # time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
        distance_diff = np.linalg.norm(locations[i] - corrections[i-1])

        prev_distance_diff = np.linalg.norm(locations[i-1] - corrections[i-1]) * time_diff

        scaler = time_diff * 10

        if distance_diff > scaler:
            # update ema
            correction_df = pd.DataFrame(corrections, columns=['x', 'y'])
            ema = correction_df.ewm(span=ema_window, adjust=False).mean()
            ema = ema[['x', 'y']].values

            # calculate the 10m circle around the previous point
            circle = np.array([corrections[i-1] + np.array([scaler * np.cos(theta), scaler * np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, 100)]) # mostly used for plotting
            correct_circle = np.array([corrections[i-1] + np.array([prev_distance_diff * np.cos(theta), prev_distance_diff * np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, 100)]) # calc new distanace
            closest_point = min(correct_circle, key=lambda point: np.linalg.norm(point - ema[i-1]))

            corrections.append(closest_point)
            """quickly plot the correction for testing"""
            # plt.plot(corrections[i-1][0], corrections[i-1][1], 'yo', label='Prev Point')
            # plt.plot(locations[i][0], locations[i][1], 'bo', label='Current Point')
            # plt.plot(circle[:, 0], circle[:, 1], label='Impossible Circle')
            # plt.plot(ema[i-1][0], ema[i-1][1], 'ro', label='EMA Point')
            # plt.plot(closest_point[0], closest_point[1], 'go', label='Corrected Point (what is added)')
            # plt.legend()
            # plt.show()
            # plt.close()
            corrected += 1
            total += 1

        else:
            total += 1
            corrections.append(locations[i])

    print(f"Corrected {corrected} out of {total} points")
    return np.array(corrections)

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

def removeOutliers_ts(df, window_time='800ms', residual_variance_threshold=0.5):
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
            fitted_line2 = model2.predict(x) # fitted_line2 = y of model2.fit
            return fitted_line2
        else:
            return fitted_line1
    
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
    def replace_fit1(y):
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
            fitted_line2 = model2.predict(x) # fitted_line2 = y of model2.fit
            return fitted_line2
        else:
            return replace_with_og(window)
        
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
                replacement_vals = replace_with_fit2(replace_fit1(window), window) 
                adjusted_index = adjusted_col_data.loc[start_time:end_time].index
                adjusted_col_data.loc[adjusted_index] = replacement_vals

        df[column] = adjusted_col_data


    # Timestamps are indexed, need to be back in forms of cols
    df.reset_index(inplace=True)
    df = df.loc[:, :'za']

    df.to_csv(f"processedRemoveOutliers.csv", index=False)
    return df


def velocityClamping_noplot(df):
    """
    Corrects the distance data in the dataframe using the data from the compass
    If the player moves more than 10m away from the beacon in a second then the data is incorrect and should be corrected
    """   
    df = df.copy()  

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')

    dt = df['timestamp'].diff().dt.total_seconds()

    for column in df.columns:
        if not column.startswith('b'):
            continue
        
        outlier_count = 1000000
        for i in range(3):
            outlier_count = mark_velocity_outliers(df, column, max_velocity=11)
            if outlier_count <= 1:
                break
            
    for column in df.columns:
        if column.startswith('b'):
            df[column].interpolate(method='linear', inplace=True)

    return df


def mark_velocity_outliers_noplot(df, column, max_velocity=11):
    outlier_count = 0

    last_valid_idx = 0
    last_valid_value = df.loc[df.index[0], column]
    last_valid_ts = df.loc[df.index[0], 'timestamp']

    for i in range(1, len(df)):
        curr_value = df.loc[df.index[i], column]
        curr_ts = df.loc[df.index[i], 'timestamp']
        
        # If current is NaN, skip updating last valid values
        if pd.isna(curr_value):
            continue
        
        dt = (curr_ts - last_valid_ts).total_seconds()
        if dt > 0 and last_valid_value is not None:
            velocity = abs(curr_value - last_valid_value) / dt
            if velocity > max_velocity:
                df.loc[df.index[i], column] = np.nan
                outlier_count += 1
                # Do NOT update last_valid_* because current value is not valid.
                continue
            else:
                # Update last valid values because this point is valid.
                last_valid_idx = i
                last_valid_value = curr_value
                last_valid_ts = curr_ts
        else:
            # If dt==0 or something is off, update the references.
            last_valid_idx = i
            last_valid_value = curr_value
            last_valid_ts = curr_ts
    
    return outlier_count

import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def circle_intersections(p1, r1, p2, r2):
    """Finds intersection points between two circles."""
    d = np.linalg.norm(p2 - p1)
    if d > r1 + r2 or d < abs(r1 - r2):  # No intersection
        return []
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2))
    p_mid = p1 + a * (p2 - p1) / d
    offset = h * np.array([-(p2[1] - p1[1]) / d, (p2[0] - p1[0]) / d])
    return [p_mid + offset, p_mid - offset]

def mark_velocity_outliers(df, column, max_velocity=11):
    """
    Scans through 'column' row by row, computing velocity from the last valid (non-NaN) point.
    If velocity > max_velocity, marks the current row as NaN.
    
    Returns:
        outlier_count: number of points marked as outliers in this pass.
        outlier_indices: list of indices (from df.index) that were marked as outliers.
        outlier_values: dict mapping those indices to the original measurement value.
    """
    outlier_count = 0
    outlier_indices = []
    outlier_values = {}
    
    # Initialize with the first row as the last valid point.
    last_valid_value = df.loc[df.index[0], column]
    last_valid_ts = df.loc[df.index[0], 'timestamp']

    for i in range(1, len(df)):
        curr_value = df.loc[df.index[i], column]
        curr_ts = df.loc[df.index[i], 'timestamp']
        
        # If current value is NaN, skip updating last valid values.
        if pd.isna(curr_value):
            continue
        
        dt = (curr_ts - last_valid_ts).total_seconds()
        if dt > 0 and last_valid_value is not None:
            velocity = abs(curr_value - last_valid_value) / dt
            if velocity > max_velocity:
                df.loc[df.index[i], column] = np.nan  # Mark outlier
                outlier_count += 1
                outlier_indices.append(df.index[i])
                outlier_values[df.index[i]] = curr_value  # Save original value
                #Do not update last_valid_* because current value is invalid.
                continue
            else:
                #This point is valid; update last valid references.
                last_valid_value = curr_value
                last_valid_ts = curr_ts
        else:
            last_valid_value = curr_value
            last_valid_ts = curr_ts
    
    return outlier_count, outlier_indices, outlier_values


def velocityClamping(df, max_speed=11, max_passes=10, plotting=False):
    """
    Corrects the distance data in the dataframe by marking outliers based on velocity.
    In each pass, it computes the velocity from the last valid (non-NaN) measurement, marks
    points with velocity > max_velocity as NaN, and then plots the current data along with red
    crosses at the locations of the outliers (using their original values).
    After all passes, a final interpolation fills the NaN values.
    """
    df = df.copy()  
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    
    for column in df.columns:
        if not column.startswith('b'):
            continue
        
        for pass_idx in range(max_passes):
            outlier_count, outlier_indices, outlier_values = mark_velocity_outliers(df, column, max_velocity=max_speed)
            print(f"Pass {pass_idx} for {column}: {outlier_count} outliers")
            
            if plotting:
                # Plot the current state of the data for this column
                plt.figure(figsize=(10,4))
                if outlier_indices:
                    # Plot red crosses at the outlier positions, using the original values stored in outlier_values
                    outlier_times = []
                    outlier_vals = []
                    for idx in outlier_indices:
                        outlier_times.append(df.loc[idx, 'timestamp'])
                        outlier_vals.append(outlier_values[idx])
                    plt.plot(outlier_times, outlier_vals, 'rx', markersize=5, label='Outliers')
                plt.plot(df['timestamp'], df[column], 'k.-', label='Data')
                plt.title(f"{column} - Pass {pass_idx} ({outlier_count} outliers)")
                plt.xlabel('Timestamp')
                plt.ylabel(column)
                plt.legend()
                plt.show()
            
            if outlier_count <= 1:
                break

        # After all passes, do a single interpolation to fill the NaN values for each beacon column.
        # for column in df.columns:
        #     if column.startswith('b'):
        #         df[column].interpolate(method='linear', inplace=True)

    df = remove_small_groups(df, threshold=20, plotting=False)
    return df


def remove_small_groups(df, threshold=5, plotting=False):
    for column in df.columns:
        if not column.startswith('b'):
            continue
        
        distances = df[column]
        result = distances.copy()
        group_start = None
        first_group = True
        removed_indices = []
        for i in range(len(distances)):
            
            #mark the start of a group
            if not np.isnan(distances[i]) and not first_group:
                if group_start is None:
                    group_start = i

            #end of a cluster
            else:
                first_group = False
                if group_start is not None:
                    length = i-group_start
                    if length <= threshold:
                        result.iloc[group_start:i] = np.nan
                        removed_indices.extend(range(group_start, i))
                    #we reached the end of a group, so we reset the counter
                    group_start = None

        #check if we end on a valid segment
        if group_start is not None:
            length = len(distances) - group_start
            if length <= threshold:
                result.iloc[group_start:len(distances)] = np.nan
                removed_indices.extend(range(group_start, len(distances)))

        df[column] = result

        if plotting:
            plt.figure(figsize=(10, 4))
            # Plot the filtered data as a black line with markers.
            plt.plot(df['timestamp'], result, 'k.-', label='Filtered Data')
            # Plot the removed points, using their original values from distances.
            if removed_indices:
                removed_times = df['timestamp'].iloc[removed_indices]
                removed_vals = distances.iloc[removed_indices]
                plt.plot(removed_times, removed_vals, 'rx', markersize=5, label='Removed Data')
            plt.title(f"{column} - Removed Small Groups (threshold = {threshold})")
            plt.xlabel("Timestamp")
            plt.ylabel(column)
            plt.legend()
            plt.show()


    return df


def velocityClamping_old(df, plotting=True, max_speed=11):
    """
    Corrects the distance data in the dataframe using the data from the compass
    If the player moves more than 10m away from the beacon in a second then the data is incorrect and should be corrected
    """   
    df = df.copy()  

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    dt = df['timestamp'].diff().dt.total_seconds()
    for column in df.columns:
        if not column.startswith('b'):
            continue
        
        outlier_count = 1000000
        for i in range(10):
            if outlier_count <= 1:
                break
            # Calculate the difference between consecutive measurements
            df[f'{column}_vel'] = abs(df[column].diff()/dt)

            # Identify the outliers where the difference is greater than 10 meters
            outliers = df[f'{column}_vel'] > max_speed
            outlier_indices = df.index[outliers] 
            # Store the original (timestamp, value) before we set them to NaN
            outlier_times = df.loc[outlier_indices, 'timestamp'].values
            outlier_values = df.loc[outlier_indices, column].values
            outlier_count = len(outlier_indices)
            print(f"Datapoints in {column}: ", len(df[column]))
            print(f"distance corrections for {column}: ", outlier_count)

            if plotting:
                # Plot the current state of the data for this column
                plt.figure(figsize=(10,4))
                plt.plot(df['timestamp'], df[column], 'k.-', label='Data')
                if not outlier_indices.empty:
                    # Plot red crosses at the outlier positions, using the original values stored in outlier_values
                    plt.plot(outlier_times, outlier_values, 'rx', markersize=5, label='Outliers')
                plt.title(f"{column} - Pass {i} ({outlier_count} outliers)")
                plt.xlabel('Timestamp')
                plt.ylabel(column)
                plt.legend()
                plt.show()

            # Replace outliers with NaN
            df.loc[outliers, column] = np.nan

            # Interpolate to fill NaN values
            df[column].interpolate(method='linear', inplace=True)

            # Drop the temporary diff column
            df.drop(columns=[f'{column}_vel'], inplace=True)  

    return df
    

# need to replace with the actual kalman in Final-kalman
# Need to replace with acc-bound
def kalmanFilter(df, x=np.array([10.0, 0]), P=np.diag([30, 16]), R=np.array([[5.]]), Q=Q_discrete_white_noise(dim=2, dt=0.3, var=2.35), dt=0.3):
    """
    Applies the Kalman filter to every column in the dataframe that starts with 'b'.
    """
    def pos_vel_filter(x, P, R, Q=0., dt=1.):
        """ Returns a KalmanFilter which implements a
        constant velocity model for a state [x dx].T
        """

        #the ones that are explicitly set in this fcn are probably less likely to be adjusted
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([x[0], x[1]]) # location and velocity
        kf.F = np.array([[1., dt],
                        [0.,  1.]])  # state transition matrix
        kf.H = np.array([[1., 0]])    # Measurement function
        kf.R *= R                     # measurement uncertainty
        if np.isscalar(P):
            kf.P *= P                 # covariance matrix 
        else:
            kf.P[:] = P               # [:] makes deep copy
        if np.isscalar(Q):
            kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
        else:
            kf.Q[:] = Q
        return kf
    
    def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0, 
        track=None, zs=None,
        count=0, do_plot=True, **kwargs):
        """
        track is the actual position of the dog, zs are the 
        corresponding measurements. 
        """

        # Simulate dog if no data provided. 
        if zs is None:
            print("no data provided, cannot run filter")
            return False

        # create the Kalman filter
        kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)  

        # run the kalman filter and store the results
        xs, cov = [], []
        for z in zs:
            kf.predict() #predicts next position
            kf.update(z) #takes next measurement, updates position
            xs.append(kf.x) #stores result
            cov.append(kf.P) #stores result

        xs, cov = np.array(xs), np.array(cov)
        return xs, cov

    df = df.copy()
    results = {}
    for column in df.columns:
        if column.startswith('b'):
            zs = df[column].values
            xs, cov = run(x, P, R, Q, dt=dt, zs=zs)
            results[column] = xs[:, 0]  # store the position estimates

    # Add the results to the dataframe, replacing the original data
    for column, values in results.items():
        df[column] = values

    return df

# Plots the abolsute error of the measurements compared to the ground truth
def absError(groundtruth, measurements, title, plot=False):
    gt = groundtruth.copy()
    measure = measurements.copy()

    filtered_data, errors = calculate_abs_error(gt, measure)

    plot_abs_error(filtered_data['timestamp'], errors, title, plot=plot)
    plot_mean_abs_error(filtered_data['timestamp'], filtered_data['mean_abs_error'], title, plot=plot)
    
    return filtered_data

def distanceCorrection(df):
    df = df.copy()  
    for column in df.columns:
        if not column.startswith('b'):
            continue
        df[column] = df[column]-0.8
    return df

def processData(filename, tests):
    # Load initial DF
    initalDf = loadData(os.path.join(os.getcwd(), filename))
    # Check if the first row has the required headers
    required_headers = ["timestamp", "wearabletimestamp", "b1d", "b2d", "b3d", "b4d", "xa", "ya", "za"]
    if list(initalDf.columns) != required_headers:
        # Add the required headers
        initalDf.columns = required_headers

    #remove error messages from data
    initalDf = cleanup_file(initalDf)


    # Ensure all values are floats/ints and not strings
    for column in initalDf.columns:
        if initalDf[column].dtype == 'object':
            try:
                initalDf[column] = initalDf[column].astype(float)
            except ValueError:
                print(f"Column {column} cannot be converted to float.")
    dfs = [initalDf]

    # Run Tests on DF
    i=0
    for testname, test in tests:
        df = dfs[-1]
        resultingDF = test(df)
        print(f"Test {testname} complete")
        resultingDF.to_csv(f"processed{testname}{i}.csv", index=False)            
        # Append the resulting DF to the list of data
        dfs.append(resultingDF)
        i+=1

    # Save all the DFS
    final = []
    final.append(("Initial", initalDf))
    i = 0
    for d in dfs[1:]:
        final.append((tests[i][0] + str(i), d))
        i += 1

    # Return a list of all the dataframes we created, final df is [-1]
    return final

def plot1d(dfs, plot=True, doc=None):
    """
    Plots 1d charts of each beacon at each step in the ppp
    """
    # create charts dictionary so we can show change over time
    charts_history = {}

    for title, df in dfs:
        # add columns to history
        for column in df.columns:
            if column.startswith('b'):
                if column not in charts_history:
                    charts_history[column] = {}
                charts_history[column][title] = df[column].values

    # Plot the history of each beacon's distance
    for beacon in charts_history:
        plt.figure(figsize=(10, 6))
        for title, data in charts_history[beacon].items():
            plt.plot(data, label=title)
        plt.xlabel('Time')
        plt.ylabel('Distance')
        plt.title(f'{beacon} Distance Over Time')
        plt.legend()
        plt.grid()
        path = os.path.join(os.getcwd(), f'charts/{beacon}_distance.png')
        plt.savefig(path)

        if doc != None: 
            add_section(doc, sectionName=f"1D {beacon}_distance", sectionText="", imgPath=path, caption=f'{beacon} Distance Over Time')
        if plot: plt.show()
        plt.close()

    return path

def find_confidence(df, beacons):
        confidence_list = []
        for _, row in df.iterrows():
            estimated_position = np.array([row['pos_x'], row['pos_y']])
            #print("pos: ", estimated_position)
            distances = np.array([row['b1d'], row['b2d'], row['b3d'], row['b4d']])
            residuals = []
            #print("# of beacons: ", len(beacons))
            for i, beacon in enumerate(beacons):
                r_i = distances[i]
                dist_est = np.linalg.norm(estimated_position - beacon)
                #print("dist est: ", dist_est)
                residuals.append((dist_est - r_i)**2)
                #print("residual: ", (dist_est - r_i)**2)

            #print("residual list: ", residuals)
            SSE = sum(residuals)
            #print("sse: ", SSE)
            alpha = 0.5
            confidence = 1.0 / (1.0 + alpha* np.sqrt(SSE))
            #print("con: ", confidence)
            confidence_list.append(confidence)

        confidence_list = np.array(confidence_list)
        return confidence_list

def plotPlayers(data, beacons, plot=True):
    """
    Plots the players' movements and 1d charts of the players' distances from each beacon, saves all plots to /charts
    """
    title = data[0]
    df = data[1]
    
    # formated like p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y ...
    finalPlayerPositions = df.copy()

    # tried predicting with ML model but didn't work great
    # def predict_with_model(input_data):
    #     # Load the model
    #     model_filepath = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/random_forest_model.pkl"
    #     model = joblib.load(model_filepath)
        
    #     # Ensure input_data is a DataFrame
    #     if not isinstance(input_data, pd.DataFrame):
    #         input_data = pd.DataFrame(input_data)
        
    #     # Make predictions
    #     predictions = model.predict(input_data)
        
    #     return predictions
  
    def trilaterate_one(beacons, distances):
        """
        Determine the position of a point using trilateration from three known points and their distances.
        
        Parameters:
        beacons: numpy array of shape (3, 2) containing the x,y coordinates of three beacons
        distances: numpy array of shape (3,) containing the distances from each beacon to the target point
        
        Returns:
        numpy array of shape (2,) containing the x,y coordinates of the calculated position
        """
        # Extract individual beacon coordinates
        P1, P2, P3 = beacons
        r1, r2, r3 = distances
        
        # Calculate vectors between points
        P21 = P2 - P1
        P31 = P3 - P1
        
        # Create coefficients matrix A and vector b for the equation Ax = b
        A = 2 * np.array([
            [P21[0], P21[1]],
            [P31[0], P31[1]]
        ])
        
        b = np.array([
            r1*r1 - r2*r2 - np.dot(P1, P1) + np.dot(P2, P2),
            r1*r1 - r3*r3 - np.dot(P1, P1) + np.dot(P3, P3)
        ])
        
        # Solve the system of equations
        try:
            position = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            raise ValueError("The beacons' positions don't allow for a unique solution")
        
        return position
    
    # Calculate player positions using trilateration
    # calculate the players positions based on the best trilateration data (three circle selma example, closest three circles)
    # calc based on averages of all combinations of 3 beacons
    # calc based on most likely next position based on acc data
    timestamps = df['timestamp']
    # acc_data = df[['xa', 'ya', 'za']]
    player_positions = []
    player_positions1 = []
    player_positions2 = []
    player_positions3 = []
    player_positions4 = []
    # ml_postions = []


    for index, row in df.iterrows():
        #if index == 0: continue # skip first row
        distances = np.array([row[col] for col in sorted(df.columns) if col.startswith('b')]) # div by 100 to convert to meters
        # distances = np.array()
        try:
            #calculate the position of the player based on a combo of three beacons
            position1 = trilaterate_one(beacons[[0, 1, 2]], distances[[0, 1, 2]])
            position2 = trilaterate_one(beacons[[0, 1, 3]], distances[[0, 1, 3]])
            position3 = trilaterate_one(beacons[[0, 2, 3]], distances[[0, 2, 3]])
            position4 = trilaterate_one(beacons[[1, 2, 3]], distances[[1, 2, 3]])
            avg = np.nanmean([position1, position2, position3, position4], axis=0)
            # position_row = pd.DataFrame([np.concatenate((avg, position1, position2, position3, position4))], columns=['x', 'y', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4'])
            # position_ml = predict_with_model(position_row)[0]
            # save avg, and individual positions
            player_positions.append(avg)
            player_positions1.append(position1)
            player_positions2.append(position2)
            player_positions3.append(position3)
            player_positions4.append(position4)
            # ml_postions.append(position_ml)

        except ValueError as e:
            print(f"Error at index {index}: {e}")
            player_positions.append([np.nan, np.nan])
            player_positions1.append([np.nan, np.nan])
            player_positions2.append([np.nan, np.nan])
            player_positions3.append([np.nan, np.nan])
            player_positions4.append([np.nan, np.nan])
            # ml_postions.append([np.nan, np.nan])

    # Add locations to a df and save to csv in case we want to analyze later
    #finalPlayerPositions['timestamp'] = df['timestamp'][1:]
    finalPlayerPositions['pos_x'] = [pos[0] for pos in player_positions]
    finalPlayerPositions['pos_y'] = [pos[1] for pos in player_positions]
    # finalPlayerPositions['x1'] = [pos[0] for pos in player_positions1]
    # finalPlayerPositions['y1'] = [pos[1] for pos in player_positions1]
    # finalPlayerPositions['x2'] = [pos[0] for pos in player_positions2]
    # finalPlayerPositions['y2'] = [pos[1] for pos in player_positions2]
    # finalPlayerPositions['x3'] = [pos[0] for pos in player_positions3]
    # finalPlayerPositions['y3'] = [pos[1] for pos in player_positions3]
    # finalPlayerPositions['x4'] = [pos[0] for pos in player_positions4]
    # finalPlayerPositions['y4'] = [pos[1] for pos in player_positions4]

    finalPlayerPositions['confidence'] = find_confidence(finalPlayerPositions, beacons)
    if title == "Ground Truth":
        finalPlayerPositions.to_csv(f'player_positions_{title}.csv', index=False)


    player_positions = np.array(player_positions)
    player_positions1 = np.array(player_positions1)
    player_positions2 = np.array(player_positions2)
    player_positions3 = np.array(player_positions3)
    player_positions4 = np.array(player_positions4)
    # ml_postions = np.array(ml_postions)


    df = trilaterate(df, beacons)
    #df = weighted_trilateration(df, beacons)
    # print("confidence stuff: ")
    # print("min: ", np.min(df["confidence"]))
    # print("max: ", np.max(df["confidence"]))
    # print("ave: ", np.mean(df["confidence"]))
    # print("std: ", np.std(df["confidence"]))
    if title != "Ground Truth":
        df = pipelineKalman_2d(df, beacons)
        #dfave = pipelineKalman_2d(finalPlayerPositions, beacons)
        df.to_csv(f'player_positions_{title}.csv', index=False)

    #kalman_positions = df[['pos_x', 'pos_y']].to_numpy()
    #corrected_positions = np.array(twoD_correction(kalman_positions.copy(), timestamps, 0))
    # Plot player positions
    plt.figure(figsize=(10, 6))
    for i in range(len(player_positions1)):
        alpha = (i + 1) / len(player_positions1)
        # plt.plot(player_positions1[i:i+2, 0], player_positions1[i:i+2, 1], 'o-', alpha=alpha, color='grey')
        # plt.plot(player_positions2[i:i+2, 0], player_positions2[i:i+2, 1], 'o-', alpha=alpha, color='green')
        # plt.plot(player_positions3[i:i+2, 0], player_positions3[i:i+2, 1], 'o-', alpha=alpha, color='purple')
        # plt.plot(player_positions4[i:i+2, 0], player_positions4[i:i+2, 1], 'o-', alpha=alpha, color='orange')
        #plt.plot(player_positions[i:i+2, 0], player_positions[i:i+2, 1], '.-', alpha=alpha, color='blue') # plot the avg last
    
        #plt.plot(corrected_positions[i:i+2, 0], corrected_positions[i:i+2, 1], 'o-', alpha=alpha, color='red')
    
    #plt.plot(df['pos_x'], df['pos_y'], '.-', alpha=alpha)

    if title != "Ground Truth":
        plt.plot(df['pos_x'], df['pos_y'], '.-', label='player trace')
        #plt.plot(dfave['pos_x'], dfave['pos_y'], '.-', color='orange', label='kalman and ave trilat')
    else:
        plt.plot(player_positions[:,0], player_positions[:,1], '.-', alpha=alpha, color='blue') # plot the avg last
    # plt.legend(['Player Path 1', 'Player Path 2', 'Player Path 3', 'Player Path 4', 'Player Path', 'New Trilateration', 'Final (Corrected) Player Path'])
    plt.scatter(beacons[:, 0], beacons[:, 1], c='red', marker='x', label='Beacons')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Player Movement Path | {title}')
    plt.legend()
    plt.grid()
    # plt.xlim(beacons[:, 0].min() - 5, beacons[:, 0].max() + 5)
    # plt.ylim(beacons[:, 1].min() - 5, beacons[:, 1].max() + 5)
    path = os.path.join(os.getcwd(), f'charts/{title}_path.png')
    plt.savefig(path)
    if plot: plt.show()
    plt.close()

    return path



def main():

    # clear charts
    for f in os.listdir(os.path.join(os.getcwd(), 'charts')):
        os.remove(os.path.join(os.getcwd(), 'charts', f))

    # Process the data
    # Submit the tests we want to run on our data in order [("testName", testFunction)]
    # ("Distance Correction", distanceCorrection)
    # ("EMA", smoothData)
    # ("Kalman Filter", pipelineKalman) - this is the right one
    # ("Kalman Filter", kalmanFilter)
    # ("Outlier Removal", removeOutliers) #this is the right one
    # ("Outlier Removal", removeOutliers_dp)
    # ("Outlier Removal", removeOutliers_ts)
    # ("Velocity Clamping", velocityClamping)
    # ("Plot", plotPlayers)
    tests = [("Distance Correction", distanceCorrection), ("Outlier Removal", removeOutliers_ts), ("Velocity Clamping", velocityClamping), ("EMA", smoothData), ("Velocity Clamping", velocityClamping)]
    #tests = [("Distance Correction", distanceCorrection), ("Velocity Clamping", velocityClamping)]
    filenames = ["obstacletest-uwb.csv"]
    gt_filename = "obstacletest-groundtruth.csv"
    # show  plots or not?
    show_plots = False
    # output doc as pdf?
    pdf = False

    for name in filenames:
        
        # start report
        doc = Document()
        gen_title(doc, author=name)

        #csv_filename = f"/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/data/{name}"
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        csv_filename = os.path.join(script_dir, "data", name)
        dfs = processData(csv_filename, tests)

        # Plot the 1d charts
        imgPath = plot1d(dfs, plot=show_plots, doc=doc)
        
        # Compare to GT Data
        gt_path = os.path.join(script_dir, "data", gt_filename)
        gt = loadData(gt_path)
        i = 0
        for df in dfs:
            print(f"\nAnalyzing {df[0]}")
            imgPath, text = GroundTruthPipeline(df[1], gt, title=df[0], plot=show_plots)
            add_section(doc, sectionName=f"{df[0]} - Ground Truth Comp.", sectionText=text, imgPath=imgPath, caption=f"{df[0]} Measured vs GT Distance", imgwidth=0.7) # image width needs to be lower fo rGT so it fits on page
            absError(gt, df[1], title=df[0], plot=show_plots)
            i += 1

        #TEMPORARY CODE TO SAVE A USEFUL CSV FOR TRILATERATION TESTING
        # i=0
        # for df in dfs:
        #     df[1].to_csv(f"processedtest{df[0]}.csv", index=False)
        #     i+=1

        # Plot GT 2d Data
        # beaconPositions = np.array([[20, 0], [0, 0], [0, 40], [20, 40]])
        #beaconPositions = np.array([[15, 0], [15, 20], [0, 0], [0, 20]])
        #beaconPositions = np.array([[0, 0], [15, 0], [0, 20], [15, 20]])
        beaconPositions = np.array([[0, 0], [12, 0], [0, 18], [12, 18]])
        #beaconPositions = np.array([[0, 0], [28.7, 0], [28.7, 25.7], [0, 25.7]])  
        imgPath = plotPlayers(("Ground Truth", gt), beaconPositions, plot=False)
        add_section(doc, sectionName="Ground Truth", sectionText="", imgPath=imgPath, caption="Ground Truth Player Movement Path")

        # Plot the final DFs
        for d in dfs:
            d[1].to_csv("processedtest.csv", index=False)
            imgPath = plotPlayers(d, beaconPositions, plot=show_plots)
            add_section(doc, sectionName=d[0], sectionText="", imgPath=imgPath, caption="Player Movement Path")

        if pdf:
            gen_pdf(doc, name.split("/")[-1]+"_report")

    
if __name__ == "__main__":
    main()
