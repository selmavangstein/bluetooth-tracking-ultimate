
"""Corrrects the inherent distance error in the data.
The distance error is assumed to be 0.8m, so this function subtracts 0.8 from all distance columns.
"""
import pandas as pd

def distanceCorrection(df):
    df = df.copy()  
    for column in df.columns:
        if not column.startswith('b'):
            continue
        df[column] = df[column]-0.8
    return df
