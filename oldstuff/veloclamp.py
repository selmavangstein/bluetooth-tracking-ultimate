"""Unused in pipeline"""


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
