
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    last_valid_idx = 0
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
                # Do not update last_valid_* because current value is invalid.
                continue
            else:
                # This point is valid; update last valid references.
                last_valid_idx = i
                last_valid_value = curr_value
                last_valid_ts = curr_ts
        else:
            last_valid_idx = i
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
                    plt.plot(outlier_times, outlier_vals, 'rx-', markersize=5, label='Outliers')
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
    df = remove_small_groups(df)
            
    return df

def remove_small_groups(df, threshold=5):
    for column in df.columns:
        if not column.startswith('b'):
            continue
        
        distances = df[column]
        result = distances.copy()
        group_start = None
        first_group = True
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
                        result[group_start:i] = np.nan
                    #we reached the end of a group, so we reset the counter
                    group_start = None

        #check if we end on a valid segment
        if group_start is not None:
            length = len(distances) - group_start
            if length <= threshold:
                result[group_start:len(distances)] = np.nan

        df[column] = result

    return df

