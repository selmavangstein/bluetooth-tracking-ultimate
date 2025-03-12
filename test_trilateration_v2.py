import numpy as np
import itertools
from sklearn.cluster import KMeans

"""CAN BE DELETED"""

"""
This file aims to go from distance measurements to position coordinates. The algorithm searches for clusters
of circle intersections, find their centroids, then take a weighted average based on the cluster spread
to find the final coordinate.
This method is not in use because it is essentially a bad way to find a coordinate. 
"""

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

import numpy as np
import itertools

def find_best_intersections(beacons, distances):
    """
    Simplified version:
    Computes and returns all intersections from every pair of beacons.
    
    Args:
        beacons (np.ndarray): Array of beacon coordinates (N, 2).
        distances (np.ndarray): Array of distances from each beacon (N,).
        
    Returns:
        all_intersections (list): A list of intersection points (each a numpy array).
    """
    all_intersections = []
    # Loop over every pair of beacons (instead of triplets)
    for i, j in itertools.combinations(range(len(beacons)), 2):
        pts = circle_intersections(beacons[i], distances[i],
                                   beacons[j], distances[j])
        all_intersections.extend(pts)
    return all_intersections


def weighted_position_from_intersections(all_intersections, beacons, max_clusters=5):
    """
    Given a list of intersection points and the beacon positions, clusters them and computes a weighted final
    position as a weighted average of the cluster centroids, where weights are the inverse of the cluster spread.
    It then checks that the chosen clusters are within the beacon bounds (with a 2-unit margin). If valid clusters
    are found, only they are used; otherwise, it falls back to using all clusters.
    
    Returns:
      weighted_final_position: The weighted final position (centroid).
      confidence: A value inversely proportional to the weighted average spread.
      labels: Cluster labels from KMeans.
      centroids: The centroids of the clusters.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    pts = np.array(all_intersections)
    if pts.shape[0] == 0:
        return None, 0.0, None, None
    if pts.shape[0] < 3:
        centroid = np.mean(pts, axis=0)
        spread = np.mean(np.linalg.norm(pts - centroid, axis=1))
        confidence = 1.0 / (spread + 1e-6)
        return centroid, confidence, np.zeros(pts.shape[0], dtype=int), np.array([centroid])
    
    # Compute bounding box from beacons (with a 2-unit margin)
    x_min, y_min = np.min(beacons, axis=0)
    x_max, y_max = np.max(beacons, axis=0)
    
    # Cluster the intersection points using KMeans.
    n_clusters = min(max_clusters, pts.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pts)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Compute the spread (average distance from the centroid) for each cluster.
    spreads = []
    for i in range(n_clusters):
        cluster_pts = pts[labels == i]
        centroid = centroids[i]
        spread = np.mean(np.linalg.norm(cluster_pts - centroid, axis=1))
        spreads.append(spread)
    spreads = np.array(spreads)
    
    # Filter clusters based on whether their centroids fall within the bounding box (with 2-unit margin).
    valid_indices = []
    for i, centroid in enumerate(centroids):
        if (x_min - 2 <= centroid[0] <= x_max + 2) and (y_min - 2 <= centroid[1] <= y_max + 2):
            valid_indices.append(i)
    
    if valid_indices:
        valid_centroids = centroids[valid_indices]
        valid_spreads = spreads[valid_indices]
    else:
        # Fallback: no cluster centroid falls within bounds, so use all clusters.
        valid_centroids = centroids
        valid_spreads = spreads
    
    # Weight clusters inversely to their spread.
    weights = 1.0 / (valid_spreads + 1e-6)
    weights /= np.sum(weights)
    weighted_final_position = np.sum(valid_centroids * weights[:, np.newaxis], axis=0)
    
    # Confidence: inverse of the weighted average spread.
    weighted_avg_spread = np.sum(valid_spreads * weights)
    confidence = 1.0 / (weighted_avg_spread + 1e-6)
    
    return weighted_final_position, confidence, labels, centroids


from sklearn.cluster import DBSCAN
import numpy as np

def weighted_position_from_intersections_DBSCAN(all_intersections, beacons, eps_start=3, eps_max=40, eps_step=2, min_samples=2, scale_factor_single_cluster = 0.5):
    """
    Clusters all intersection points using DBSCAN, tuning eps from eps_start to eps_max 
    until at least one valid cluster is found. A valid cluster has at least two points 
    and its centroid lies within a 2-unit margin around the beacon bounding box.
    
    Then, compute a weighted average of the centroids (weighted by count and inverse spread)
    to obtain the final position, and a confidence value (inverse of the weighted average spread).
    
    Returns:
      weighted_final_position: final weighted estimated position (numpy array)
      confidence: a numerical confidence value (higher means more confident)
      labels: DBSCAN labels for each point in all_intersections
      valid_centroids: numpy array of centroids for valid clusters (or None if none valid)
    """
    pts = np.array(all_intersections)
    if pts.shape[0] == 0:
        return None, 0.0, None, None
    
    # Compute bounding box from beacons (with a 2-unit margin)
    x_min, y_min = np.min(beacons, axis=0)
    x_max, y_max = np.max(beacons, axis=0)
    
    eps = eps_start
    valid_labels = []
    clusters_info = {}  # key: label, value: dict with 'centroid', 'spread', 'count'
    labels = None

    
    # Gradually increase eps until we get at least one valid cluster (or reach eps_max)
    while eps <= eps_max:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
        labels = dbscan.labels_
        unique_labels = set(labels)
        valid_labels = []
        clusters_info = {}
        
        for label in unique_labels:
            if label == -1:
                continue  # noise
            cluster_pts = pts[labels == label]
            if len(cluster_pts) < 2:
                continue  # not enough points for a valid cluster
            centroid = np.mean(cluster_pts, axis=0)
            # Check if centroid is within bounds (2-unit margin)
            if (x_min - 2 <= centroid[0] <= x_max + 2) and (y_min - 2 <= centroid[1] <= y_max + 2):
                spread = np.mean(np.linalg.norm(cluster_pts - centroid, axis=1))
                valid_labels.append(label)
                clusters_info[label] = {'centroid': centroid, 'spread': spread, 'count': len(cluster_pts)}
        
        if valid_labels:
            break
        eps += eps_step

    # If no valid clusters are found, fallback to overall mean of points.
    if not valid_labels:
        centroid = np.mean(pts, axis=0)
        spread = np.mean(np.linalg.norm(pts - centroid, axis=1))
        confidence = 1.0 / (spread + 1e-6)
        return centroid, confidence, labels, None
    
    # Gather centroids, spreads, and counts from valid clusters.
    valid_centroids = []
    valid_spreads = []
    counts = []
    for label in valid_labels:
        info = clusters_info[label]
        valid_centroids.append(info['centroid'])
        valid_spreads.append(info['spread'])
        counts.append(info['count'])
    
    valid_centroids = np.array(valid_centroids)
    valid_spreads = np.array(valid_spreads)
    counts = np.array(counts)
    
    # Weight clusters: here we use (count / spread) as weight.
    weights = counts / (valid_spreads + 1e-6)
    weights /= np.sum(weights)
    weighted_final_position = np.sum(valid_centroids * weights[:, np.newaxis], axis=0)
    weighted_avg_spread = np.sum(valid_spreads * weights)

    # If only one valid cluster, scale the spread to boost confidence.
    if len(valid_labels) == 1:
        effective_spread = weighted_avg_spread * scale_factor_single_cluster
    else:
        effective_spread = weighted_avg_spread
        
    confidence = 1.0 / (1.0 + effective_spread)
    
    return weighted_final_position, confidence, labels, valid_centroids



# Example of integrating everything:
def weighted_trilateration(df, beacon_positions):
    """
    Processes a DataFrame of beacon distance measurements.
    For each row, it computes intersections, then uses clustering on all intersections 
    to produce a weighted final position and a confidence score.
    Returns a copy of the DataFrame with additional columns for the weighted position 
    and confidence.
    """
    import pandas as pd
    # Identify distance columns (those that start with 'b')
    distance_columns = [col for col in df.columns if col.startswith('b')]
    df_result = df.copy()
    pos_x, pos_y, confidences = [], [], []

    for _, row in df.iterrows():
        distances = np.array(row[distance_columns])
        # Compute intersections using your existing algorithm.
        all_intersections = find_best_intersections(beacon_positions, distances)
        # Now use clustering on all intersections to compute a weighted final position.
        weighted_pos, confidence, labels, valid_centroids = weighted_position_from_intersections_DBSCAN(all_intersections, beacon_positions)
        if weighted_pos is None:
            pos_x.append(np.nan)
            pos_y.append(np.nan)
            confidences.append(0.0)
        else:
            pos_x.append(weighted_pos[0])
            pos_y.append(weighted_pos[1])
            confidences.append(confidence)
    df_result['pos_x'] = pos_x
    df_result['pos_y'] = pos_y
    df_result['confidence'] = confidences
    return df_result

import matplotlib.pyplot as plt
import numpy as np

def plot_results(beacons, distances, all_intersections, labels, valid_centroids, final_position, gt_pos=None):
    """
    Plots beacon circles, all intersection points (color-coded by DBSCAN cluster),
    the valid cluster centroids, and the final weighted estimated position.
    Modified to ensure clusters get distinct colors.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot beacon circles and positions.
    for i, (beacon, radius) in enumerate(zip(beacons, distances)):
        circle = plt.Circle(beacon, radius, color=f"C{i}", alpha=0.3, fill=False)
        ax.add_patch(circle)
        ax.scatter(*beacon, color=f"C{i}")
        ax.text(beacon[0], beacon[1], f"B{i+1}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
    pts = np.array(all_intersections)

    # Plot valid cluster centroids.
    if valid_centroids is not None:
        valid_centroids = np.array(valid_centroids)
        ax.scatter(valid_centroids[:, 0], valid_centroids[:, 1], color="black", marker="o", s=50, 
                   label="Valid cluster centroids")
    
    # Plot final weighted estimated position.
    if final_position is not None:
        ax.scatter(*final_position, color="red", marker="*", s=200, label="Final estimated position")
    
    # Plot intersections, color-coded by DBSCAN labels.
    if pts.size > 0 and labels is not None:
        unique_labels = sorted(set(labels))
        # Prepare a colormap for clusters (excluding noise)
        n_clusters = len([lbl for lbl in unique_labels if lbl != -1])
        cmap = plt.get_cmap("viridis", n_clusters)
        for lbl in unique_labels:
            cluster_pts = pts[labels == lbl]
            if lbl == -1:
                # Noise gets a fixed color (e.g., gray)
                color = "gray"
                label_name = "Noise"
            else:
                # Map each non-noise label to a unique color.
                # Get the index among non-noise labels:
                non_noise_labels = [l for l in unique_labels if l != -1]
                color_idx = non_noise_labels.index(lbl)
                color = cmap(color_idx)
                label_name = f"Cluster {lbl}"
            ax.scatter(cluster_pts[:, 0], cluster_pts[:, 1], color=color, marker="x", label=label_name)

    if gt_pos is not None:        
        ax.scatter(gt_pos[0], gt_pos[1], color="green", marker="+", s=50, 
                   label="ground truth")
    #ax.set_xlim(beacons[:, 0].min() - 2, beacons[:, 0].max() + 2)
    #ax.set_ylim(beacons[:, 1].min() - 2, beacons[:, 1].max() + 2)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Trilateration with DBSCAN Weighted Clustering")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()




# Example usage with test data:
import pandas as pd
if __name__=='__main__':
    testdata = pd.read_csv("processedtest6.csv")
    gt = pd.read_csv("player_positions_Ground Truth.csv")

    testdata['timestamp'] = pd.to_datetime(testdata['timestamp'])
    gt['timestamp'] = pd.to_datetime(gt['timestamp'])

    row = testdata.iloc[140]
    ts = row['timestamp']
    closest_idx = (gt['timestamp'] - ts).abs().idxmin()
    gt_row = gt.loc[closest_idx]
    gt_pos = [gt_row['x'], gt_row['y']]

    distances = np.array(row[[col for col in testdata.columns if col.startswith('b')]])
    beacons = np.array([[0, 0], [12, 0], [0, 18], [12, 18]])  # Example beacon positions

    # Get the intersections from one row (for plotting/debugging)
    all_intersections = find_best_intersections(beacons, distances)
    weighted_pos, confidence, labels, centroids = weighted_position_from_intersections_DBSCAN(all_intersections, beacons, eps_step=1)
    print("Weighted final position:", weighted_pos)
    print("Confidence:", confidence)
    print("Ground truth position:", gt_pos)

    plot_results(beacons, distances, all_intersections, labels, centroids, weighted_pos, gt_pos=gt_pos)