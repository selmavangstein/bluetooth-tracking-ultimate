import numpy as np
import itertools
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

"""
This file contains the coordinate estimation algorithm.
Its purpose is to go from a set of distance measurements to a set of coordinates representing position on the field.
Instead of trilateration, we utilize a clustering algorithm that identifies the densest cluster of circle intersections,
and uses its centroid as the final coordinate. This utilizes the data redundancy we have from four beacons.
"""

def circle_intersections(p1, r1, p2, r2):
    """
    Finds intersection points between two circles.
    Args: 
        radii and centers of the two circles.
    Output: 
        List of intersections.
        """
    d = np.linalg.norm(p2 - p1)

    if d > r1 + r2 or d < abs(r1 - r2):  #No intersection
        return []
    
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2))
    p_mid = p1 + a * (p2 - p1) / d
    offset = h * np.array([-(p2[1] - p1[1]) / d, (p2[0] - p1[0]) / d])

    return [p_mid + offset, p_mid - offset]

def find_best_intersections(beacons, distances):
    """
    Detects the densest cluster and finds its centroid. 
    Args:
        beacons:  Array of beacon coordinates.
        distances: Array of distance measurements to each beacon.
    Returns:
        estimated_position: Centroid of the densest cluster
        confidence: Value describing confidence in the final coordinate (meant to be sent to Kalman filter)
        all_intersections: List of all the circle intersections (for plotting/debugging reasons)
        best_cluster_points: The intersection points in the densest cluster (for plotting/debugging reasons)
    """
    all_intersections = []  # Store all intersection points found
    estimated_position = np.array([None, None])
    best_cluster_points = None
    distances = np.array(distances, dtype=float)

    #ignore nans
    valid_mask = ~np.isnan(distances)
    beacons = beacons[valid_mask]
    distances = distances[valid_mask]

    # Compute intersections for all unique beacon pairs
    if len(beacons)<2:
        estimated_position = np.array([100,100])
        confidence = 0
        all_intersections = np.array([])
        best_cluster_points = np.array([])
        return estimated_position, confidence, all_intersections, best_cluster_points
    
    for i, j in itertools.combinations(range(len(beacons)), 2):
        intersections = circle_intersections(beacons[i], distances[i], beacons[j], distances[j])
        all_intersections.extend(intersections)  

    
    all_intersections = np.array(all_intersections) 
    eps = 0.2  #Maximum distance between points in a cluster
    eps_max = 60 
    min_samples = 2  #Minimum points to form a cluster

    #bounds for field
    x_min, y_min = np.min(beacons, axis=0)
    x_max, y_max = np.max(beacons, axis=0)

    while np.all(estimated_position == np.array([None, None])):
        #if no clusters are found, set the position to be the mean of all intersections
        if eps > eps_max:
            estimated_position = np.mean(all_intersections, axis=0)
            best_cluster_points = all_intersections
            break

        #Identify clusters of intersections
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(all_intersections)
        labels = np.array(db.labels_)
        valid_clusters = labels[labels != -1]

        #Identify the largest cluster
        if len(valid_clusters) > 0:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            best_cluster_label = unique_labels[np.argmax(counts)]
            best_cluster_points = all_intersections[labels == best_cluster_label]

            #Compute final estimated position (mean of best cluster)
            estimated_position = best_cluster_points.mean(axis=0)

            #If the coordinate is out of bounds, try again
            if estimated_position[0] < x_min-2 or estimated_position[0] > x_max+2:
                estimated_position = np.array([None, None])
                labels[labels == best_cluster_label] = -1
                valid_clusters = labels[labels != -1]
                if eps<2:
                    eps+=0.2
                else:
                    eps +=1

            elif estimated_position[1] < y_min-2 or estimated_position[1] > y_max+2:
                estimated_position = np.array([None, None])
                labels[labels == best_cluster_label] = -1
                valid_clusters = labels[labels != -1]
                if eps<2:
                    eps+=0.2
                else:
                    eps +=1

        #If no clusters are found, try again with a larger cluster size
        else:
            if eps<2:
                eps+=0.2
            else:
                eps +=1

    #Calculate a confidence value for the final coordinate
    residuals = []
    for i, beacon in enumerate(beacons):
        r_i = distances[i]
        dist_est = np.linalg.norm(estimated_position - beacon)
        residuals.append((dist_est - r_i)**2)

    SSE = sum(residuals)
    alpha = 0.5
    confidence = 1.0 / (1.0 + alpha* np.sqrt(SSE))

    return estimated_position, confidence, all_intersections, best_cluster_points

def trilaterate(df, beacon_positions):
    """Computes player positions based on beacon distances and adds them to the DataFrame.

    Args:
        df: DataFrame containing distance measurements from beacons.
        beacon_position: Array of beacon coordinates.

    Returns:
        df: A dataframe copy of df with added 'pos_x' and 'pos_y' columns.
    """

    distance_columns = [col for col in df.columns if col.startswith('b')]
    df_result = df.copy()

    pos_x, pos_y = [], []
    confidence_list = []

    for index, row in df.iterrows():
        distances = np.array(row[distance_columns])

        #Find the coordinate for each datapoint
        try:
            final_position, confidence  = find_best_intersections(beacon_positions, distances)
            pos_x.append(final_position[0])
            pos_y.append(final_position[1])
            confidence_list.append(confidence)
            
        except ValueError:
            #If trilateration fails, append NaN
            print("Unable to identify coordinate")
            pos_x.append(np.nan)
            pos_y.append(np.nan)
            confidence_list.append(np.nan)

    #Add computed positions to the DataFrame
    df_result['pos_x'] = pos_x
    df_result['pos_y'] = pos_y
    df_result['confidence'] = confidence_list

    return df_result

