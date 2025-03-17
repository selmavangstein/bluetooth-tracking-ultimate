
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

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


def main():
    # test here
    pass

if __name__ == "__main__":
    main()
    