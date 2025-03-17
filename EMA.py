""" 
Exponential Moving Average (EMA) for smoothing data 
"""

import pandas as pd

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
