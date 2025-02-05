
"""
Inputs one csv from each player or the plain txt file from each player (and then turns that into a csv)
Outputs an animated chart of the players' movements and 1d charts of the players' distances from each beacon

The program will only process data in columns that starts with a 'b' (for beacon data) (Ex. b1d) 
Lets keep that naming convention for the beacon data columns so we can add as many as we like without changing the code

Uses a pandas df to store the data, and a matplotlib animation to animate the data
"""

from analyze_ftm_data import analyze_ftm_data
from abs_error import *

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
        
def pipelineRemoveOutliers(df, window_size=20, residual_variance_threshold=1.5, plot=True):
    """
    Removes outliers from the data and replaces them with linear fit values.
    Revamped so it would fit in the pipeline and not include GT
    """
    df = df.copy()

    # df['realtimestamp'] = pd.to_datetime(df['realtimestamp'], format='%H:%M:%S.%f')

    pos_data = df['b3d']  # Possible columns: d3, d2, d1, d4 

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
                adjusted_pos_data.iloc[j] = min(replacement_values[curr_window_index], pos_data.iloc[j])

    # Probably can get rid of this 
    # plt.figure()
    # plt.plot(df['realtimestamp'], pos_data, label="Measurement")
    # plt.plot(df['realtimestamp'], adjusted_pos_data, label="Obstacle Replaced")
    # plt.title(f"{title} Measurement with Obstacle Replacement")
    # plt.legend()
    # plt.savefig(os.path.join(os.getcwd(), f'charts/{title}_path.png'))
    # if plot: plt.show()
    # plt.close()

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

def distanceCorrection(df):
    """
    Corrects the distance data in the dataframe using the data from the compass
    If the player moves more than 10m away from the beacon in a second then the data is incorrect and should be corrected
    """
    df = df.copy()  

    for column in df.columns:
        if not column.startswith('b'):
            continue

        # Calculate the difference between consecutive measurements
        df[f'{column}_diff'] = df[column].diff().abs()

        # Identify the outliers where the difference is greater than 10 meters
        outliers = df[f'{column}_diff'] > 9

        # Replace outliers with NaN
        df.loc[outliers, column] = np.nan

        # Interpolate to fill NaN values
        df[column].interpolate(method='linear', inplace=True)

        # Drop the temporary diff column
        df.drop(columns=[f'{column}_diff'], inplace=True)

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
def absError(measurements, title="", gt="jan17-groundtruth.csv", plot=False):
    measurements = measurements.copy()

    script_dir = os.path.dirname(os.path.abspath(__file__))  
 
    datafile = os.path.join(script_dir, "data", gt)  
    if not os.path.exists(datafile):
        print(f"File not found: {datafile}")
    groundtruth = pd.read_csv(datafile)

    filtered_data, mean_error = calculate_abs_error(groundtruth, measurements)

    plot_abs_error(filtered_data['timestamp'], filtered_data['abs_error'], mean_error, plot=plot, title=title)

    return filtered_data

def processData(filename, tests):
    
    # Load initial DF
    initalDf = loadData(os.path.join(os.getcwd(), filename))
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
    final.append(("Initial", initalDf))
    i = 0
    for d in dfs[1:]:
        final.append((tests[i][0] + str(i), d))
        i += 1

    # Return a list of all the dataframes we created, final df is [-1]
    return final

def plot1d(dfs, plot=True):
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
        plt.savefig(os.path.join(os.getcwd(), f'charts/{beacon}_distance.png'))
        if plot: plt.show()
        plt.close()

def plotPlayers(data, beacons, plot=True):
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
    # calculate the players positions based on the best trilateration data (three circle selma example, closest three circles)
    # calc based on averages of all combinations of 3 beacons
    # calc based on most likely next position based on acc data
    player_positions = []
    player_positions1 = []
    player_positions2 = []
    player_positions3 = []
    player_positions4 = []


    for index, row in df.iterrows():
        if index == 0: continue # skip first row
        distances = np.array([row[col] / 100 for col in df.columns if col.startswith('b')]) # div by 100 to convert to meters
        try:
            #calculate the position of the player based on a combo of three beacons
            position1 = trilaterate_one(beacons[[0, 1, 2]], distances[[0, 1, 2]])
            position2 = trilaterate_one(beacons[[0, 1, 3]], distances[[0, 1, 3]])
            position3 = trilaterate_one(beacons[[0, 2, 3]], distances[[0, 2, 3]])
            position4 = trilaterate_one(beacons[[1, 2, 3]], distances[[1, 2, 3]])
            # save avg, and individual positions
            player_positions.append(np.mean([position1, position2, position3, position4], axis=0))
            player_positions1.append(position1)
            player_positions2.append(position2)
            player_positions3.append(position3)
            player_positions4.append(position4)

        except ValueError as e:
            print(f"Error at index {index}: {e}")
            player_positions.append([np.nan, np.nan])
            player_positions1.append([np.nan, np.nan])
            player_positions2.append([np.nan, np.nan])
            player_positions3.append([np.nan, np.nan])
            player_positions4.append([np.nan, np.nan])


    player_positions = np.array(player_positions)
    player_positions1 = np.array(player_positions1)
    player_positions2 = np.array(player_positions2)
    player_positions3 = np.array(player_positions3)
    player_positions4 = np.array(player_positions4)

    # Plot player positions
    plt.figure(figsize=(10, 6))
    # plt.plot(player_positions[:, 0], player_positions[:, 1], 'o-', label='Player Path')
    plt.plot(player_positions1[:, 0], player_positions1[:, 1], 'o-', label='Player Path 1', alpha=0.5)
    # plt.plot(player_positions2[:, 0], player_positions2[:, 1], 'o-', label='Player Path 2', alpha=0.5)
    # plt.plot(player_positions3[:, 0], player_positions3[:, 1], 'o-', label='Player Path 3', alpha=0.5)
    # plt.plot(player_positions4[:, 0], player_positions4[:, 1], 'o-', label='Player Path 4', alpha=0.5)
    plt.scatter(beacons[:, 0], beacons[:, 1], c='red', marker='x', label='Beacons')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Player Movement Path | {title}')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(os.getcwd(), f'charts/{title}_path.png'))
    if plot: plt.show()
    plt.close()



def main():
    # clear charts
    for f in os.listdir(os.path.join(os.getcwd(), 'charts')):
        os.remove(os.path.join(os.getcwd(), 'charts', f))

    # Process the data
    # Submit the tests we want to run on our data in order [("testName", testFunction)]
    # ("Distance Correction", distanceCorrection)
    # ("EMA", smoothData)
    # ("Kalman Filter", kalmanFilter)
    # ("Outlier Removal", removeOutliers)
    # ("Plot", plotPlayers)
    tests = [("Outlier Removal", removeOutliers), ("Kalman Filter", kalmanFilter), ("EMA", smoothData), ("Distance Correction", distanceCorrection)]

    csv_filename = "4beaconv1.csv"
    dfs = processData(csv_filename,tests)

    # Plot the 1d charts
    plot1d(dfs, plot=False)

    # Compare to GT Data
    gt = loadData("GroundyTruthy.csv")
    for df in dfs:
        print(f"\nAnalyzing {df[0]}")
        analyze_ftm_data(df[1], gt, title=df[0], plot=False)
        absError(df[1], title=df[0], plot=False)

    # Plot the final DFs
    beaconPositions = np.array([[20, 0], [0, 0], [0, 40], [20, 40]])
    for d in dfs:
        plotPlayers(d, beaconPositions, plot=False)

    

if __name__ == "__main__":
    main()
