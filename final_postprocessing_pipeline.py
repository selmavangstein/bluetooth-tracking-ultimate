
"""
Inputs one csv from each player or the plain txt file from each player (and then turns that into a csv)
Outputs an animated chart of the players' movements and 1d charts of the players' distances from each beacon

The program will only process data in columns that starts with a 'b' (for beacon data) (Ex. b1d) 
Lets keep that naming convention for the beacon data columns so we can add as many as we like without changing the code

Uses a pandas df to store the data, and a matplotlib animation to animate/plot the data
"""

# pre-built
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.kalman import KalmanFilter
# custom
from report import gen_title, add_section, gen_pdf, Document
from cleanData import clean, loadData, is_valid_row, cleanup_file, processData
from EMA import smoothData
from outlierFunctions import removeOutliers, removeOutliers_dp, removeOutliers_ts, pipelineRemoveOutliers
from distanceCorrection import distanceCorrection
from veloClamp import mark_velocity_outliers, velocityClamping, remove_small_groups
from twoDCorrections import twoD_correction
from GroundTruthPipeline import GroundTruthPipeline
from abs_error import *
from kalman_filter_pos_vel_acc import pipelineKalman, find_confidence
from kalman_2d import pipelineKalman_2d
from final_trilateration import trilaterate
from test_trilateration_v2 import weighted_trilateration


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

def plotPlayers(data, beacons, plot=True):
    """
    Plots the players' movements and 1d charts of the players' distances from each beacon, saves all plots to /charts
    """
    title = data[0]
    df = data[1]

    # remove nans from df
    df = df.dropna()
    
    # formated like p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y ...
    finalPlayerPositions = df.copy()
  
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
    timestamps = df['timestamp']
    player_positions = []
    player_positions1 = []
    player_positions2 = []
    player_positions3 = []
    player_positions4 = []

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
            # save avg, and individual positions
            player_positions.append(avg)
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

    # Add locations to a df and save to csv in case we want to analyze later
    finalPlayerPositions['pos_x'] = [float(pos[0]) for pos in player_positions]
    finalPlayerPositions['pos_y'] = [float(pos[1]) for pos in player_positions]
    # finalPlayerPositions['x1'] = [pos[0] for pos in player_positions1]
    # finalPlayerPositions['y1'] = [pos[1] for pos in player_positions1]
    # finalPlayerPositions['x2'] = [pos[0] for pos in player_positions2]
    # finalPlayerPositions['y2'] = [pos[1] for pos in player_positions2]
    # finalPlayerPositions['x3'] = [pos[0] for pos in player_positions3]
    # finalPlayerPositions['y3'] = [pos[1] for pos in player_positions3]
    # finalPlayerPositions['x4'] = [pos[0] for pos in player_positions4]
    # finalPlayerPositions['y4'] = [pos[1] for pos in player_positions4]

    # add confidence for position guess
    finalPlayerPositions['confidence'] = find_confidence(finalPlayerPositions, beacons)

    # saves a csv if needed
    # finalPlayerPositions.to_csv(f'player_positions_{title}.csv', index=False)

    player_positions = np.array(player_positions)
    player_positions1 = np.array(player_positions1)
    player_positions2 = np.array(player_positions2)
    player_positions3 = np.array(player_positions3)
    player_positions4 = np.array(player_positions4)

    """ERROR HERE"""
    # finalPlayerPositions = trilaterate(finalPlayerPositions, beacons) ## another way to calculate the player positions
    df = weighted_trilateration(finalPlayerPositions, beacons) 

    # print("confidence stuff: ")
    # print("min: ", np.min(df["confidence"]))
    # print("max: ", np.max(df["confidence"]))
    # print("ave: ", np.mean(df["confidence"]))
    # print("std: ", np.std(df["confidence"]))
    # if "Ground Truth" not in title:
    #     finalPlayerPositions = pipelineKalman_2d(finalPlayerPositions, beacons)

    kalman_positions = finalPlayerPositions[['pos_x', 'pos_y']].to_numpy()
    corrected_positions = np.array(twoD_correction(kalman_positions.copy(), timestamps, 0))

    # Plot player positions
    plt.figure(figsize=(10, 6))
    for i in range(len(player_positions1)):
        alpha = (i + 1) / len(player_positions1)
        # option to plot different beacon groupings
        # plt.plot(player_positions1[i:i+2, 0], player_positions1[i:i+2, 1], 'o-', alpha=alpha, color='grey')
        # plt.plot(player_positions2[i:i+2, 0], player_positions2[i:i+2, 1], 'o-', alpha=alpha, color='green')
        # plt.plot(player_positions3[i:i+2, 0], player_positions3[i:i+2, 1], 'o-', alpha=alpha, color='purple')
        # plt.plot(player_positions4[i:i+2, 0], player_positions4[i:i+2, 1], 'o-', alpha=alpha, color='orange')
        # plot the avg
        plt.plot(player_positions[i:i+2, 0], player_positions[i:i+2, 1], '.-', alpha=alpha, color='blue') 
        # plot the corrected 2d kalman
        plt.plot(corrected_positions[i:i+2, 0], corrected_positions[i:i+2, 1], 'o-', alpha=alpha, color='red') 
    
    plt.plot(finalPlayerPositions['pos_x'], finalPlayerPositions['pos_y'], '.-', alpha=alpha)

    if title != "Ground Truth":
        plt.plot(finalPlayerPositions['pos_x'], finalPlayerPositions['pos_y'], '.-', label='kalman and ave cluster')
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
    # in case we need to limit the graph bounds
    # plt.xlim(beacons[:, 0].min() - 5, beacons[:, 0].max() + 5)
    # plt.ylim(beacons[:, 1].min() - 5, beacons[:, 1].max() + 5)
    path = os.path.join(os.getcwd(), f'charts/{title}_path.png')
    plt.savefig(path)
    if plot: plt.show()
    plt.close()

    # Create an animated version of the player movement path
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(beacons[:, 0].min() - 5, beacons[:, 0].max() + 5)
    ax.set_ylim(beacons[:, 1].min() - 5, beacons[:, 1].max() + 5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Player Movement Path | {title}')
    ax.scatter(beacons[:, 0], beacons[:, 1], c='red', marker='x', label='Beacons')

    # Instead of a single line, we'll use:
    # 1. A trail of past points (low alpha)
    # 2. A current point (full alpha)
    past_points, = ax.plot([], [], 'o-', color='blue', alpha=0.05, label='Path History')
    current_point, = ax.plot([], [], 'o', color='red', markersize=8, label='Current Point')

    ax.legend()
    ax.grid()

    def init():
        past_points.set_data([], [])
        current_point.set_data([], [])
        return past_points, current_point

    def update(frame):
        # Plot path history with low alpha
        if frame > 0:  # Only if we have past points
            past_points.set_data(player_positions[:frame, 0], player_positions[:frame, 1])
        else:
            past_points.set_data([], [])
        
        # Plot current point with full alpha
        current_point.set_data([player_positions[frame, 0]], [player_positions[frame, 1]])
        
        return past_points, current_point

    ani = FuncAnimation(fig, update, frames=len(player_positions), init_func=init, blit=True, repeat=False, interval=50)
    anim_path = os.path.join(os.getcwd(), f'charts/{title}_path_animation.mp4')
    ani.save(anim_path, writer='ffmpeg', fps=50)

    if plot:
        plt.show()
    plt.close()

    return path, anim_path

def main():
    # clear charts from storage so it doesn't get cluttered
    for f in os.listdir(os.path.join(os.getcwd(), 'charts')):
        os.remove(os.path.join(os.getcwd(), 'charts', f))

    # Process the 1d data
    # Submit the tests we want to run on our data in order [("testName", testFunction)]
    # ("Distance Correction", distanceCorrection)
    # ("EMA", smoothData)
    # ("Kalman Filter", pipelineKalman) - this is the right one
    # ("Outlier Removal", removeOutliers) # this is the right one
    # ("Outlier Removal", removeOutliers_dp)
    # ("Outlier Removal", removeOutliers_ts)
    tests = [("Distance Correction", distanceCorrection), ("Velocity Clamping", velocityClamping), ("Outlier Removal", removeOutliers), ("Kalman Filter", pipelineKalman), ("EMA", smoothData), ("Velocity Clamping", velocityClamping)]
    filenames = ["feb23/aroundsquare-uwb.csv"]
    gt_filename = "feb23/aroundsquare-groundtruth.csv"
    
    """
    Options
    """
    # show  plots or not?
    # plots auto save the /charts so no need to show them (gets annoying)
    show_plots = False
    # output doc as pdf?
    pdf = False

    # I don't recommend doing it on multiple files but the functionallity is there
    for name in filenames:
        # start report
        doc = Document()
        if pdf:
            gen_title(doc, author=name)

        # create dataframe
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        csv_filename = os.path.join(script_dir, "data", name)
        dfs = processData(csv_filename, tests)

        # Plot the 1d charts
        imgPath = plot1d(dfs, plot=show_plots, doc=doc)
        
        # Compare to GT Data
        gt_path = os.path.join(script_dir, "data", gt_filename)
        gt = loadData(gt_path)
        i = 0
        for df in dfs.copy():
            print(f"\nAnalyzing {df[0]}")
            imgPath, text = GroundTruthPipeline(df[1], gt, title=df[0], plot=show_plots)
            if pdf: add_section(doc, sectionName=f"{df[0]} - Ground Truth Comp.", sectionText=text, imgPath=imgPath, caption=f"{df[0]} Measured vs GT Distance", imgwidth=0.7) # image width needs to be lower fo rGT so it fits on page
            absError(gt, df[1], title=df[0], plot=show_plots)
            i += 1

        """
        TEMPORARY CODE TO SAVE A USEFUL CSV FOR TRILATERATION TESTING
        i=0
        for df in dfs:
            df[1].to_csv(f"processedtest{i}.csv", index=False)
            i+=1 
        """

        # Plot GT 2d Data
        # FORMAT 
        # beaconPositions = np.array([[20, 0], [0, 0], [0, 40], [20, 40]])
        beaconPositions = np.array([[0, 0], [28.7, 0], [28.7, 25.7], [0, 25.7]])  
        imgPath = plotPlayers(("Ground Truth", gt), beaconPositions, plot=False)[0]
        add_section(doc, sectionName="Ground Truth", sectionText="", imgPath=imgPath, caption="Ground Truth Player Movement Path")

        # Plot the final DFs
        for d in dfs:
            # save to csv if needed
            # d[1].to_csv("processedtest.csv", index=False)
            imgPath = plotPlayers(d, beaconPositions, plot=show_plots)
            if pdf: add_section(doc, sectionName=d[0], sectionText="", imgPath=imgPath, caption="Player Movement Path")

        if pdf:
            gen_pdf(doc, name.split("/")[-1]+"_report")


if __name__ == "__main__":
    main()
