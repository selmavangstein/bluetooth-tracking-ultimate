
"""
Inputs one csv from each player or the plain txt file from each player (and then turns that into a csv)
Outputs an animated chart of the players' movements and 1d charts of the players' distances from each beacon

The program will only process data in columns that starts with a 'b' (for beacon data) (Ex. b1d) 
Lets keep that naming convention for the beacon data columns so we can add as many as we like without changing the code

Uses a pandas df to store the data, and a matplotlib animation to animate the data
"""

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

import matplotlib.pyplot as plt

def loadData(filename):
    """
    Loads the data from a csv file into a pandas dataframe
    """
    return pd.read_csv(filename)

def smoothData(df, window_size=25):
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

def processData():
    # Submit the tests we want to run on our data in order [("testName", testFunction)]
    # ("EMA", smoothData)
    # ("Kalman Filter", kalmanFilter)
    # ("Outlier Removal", removeOutliers)
    # ("Plot", plotPlayers)
    tests = [("EMA", smoothData)]

    # Load initial DF
    initalDf = loadData("playercsvs/4beaconv1.csv")
    dfs = [initalDf]

    # Run Tests on DF
    for testname, test in tests:
        df = dfs[-1]
        resultingDF = test(df)

        print(f"Test {testname} complete")

        # Append the resulting DF to the list of data
        dfs.append(resultingDF)

    # Save all the DFS
    final = []
    i = 0
    for d in dfs[1:]:
        final.append((tests[i][0] + str(i), d))
        i += 1

    # Return a list of all the dataframes we created, final df is [-1]
    return final

def plotPlayers(data, beacons):
    """
    Plots the players' movements and 1d charts of the players' distances from each beacon, saves all plots to /charts
    """
    title = data[0]
    df = data[1]
    
    # formated like p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y ...
    finalPlayerPositions = pd.DataFrame()
  
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
    player_positions = []

    for index, row in df.iterrows():
        if index == 0: continue # skip first row
        distances = [row[col] / 100 for col in df.columns if col.startswith('b')] # div by 100 to convert to meters
        distances = distances[:-1] # remove the last column
        try:
            position = trilaterate_one(beacons, distances)
            player_positions.append(position)
        except ValueError as e:
            print(f"Error at index {index}: {e}")
            player_positions.append([np.nan, np.nan])

    player_positions = np.array(player_positions)

    # Plot player positions
    plt.figure(figsize=(10, 6))
    plt.plot(player_positions[:, 0], player_positions[:, 1], 'o-', label='Player Path')
    plt.scatter(beacons[:, 0], beacons[:, 1], c='red', marker='x', label='Beacons')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Player Movement Path | {title}')
    plt.legend()
    plt.grid()
    # plt.savefig('/charts/player_movement.png')
    plt.show()



def main():
    # Process the data
    dfs = processData()

    # Plot the final DFs
    beaconPositions = np.array([[20, 0], [0, 0], [20, 40]])
    for d in dfs:
        plotPlayers(d, beaconPositions)

if __name__ == "__main__":
    main()
