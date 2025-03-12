import numpy as np
import itertools
from sklearn.cluster import KMeans
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN

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

def circle_intersections_with_pair(p1, r1, p2, r2, pair):
    """Finds intersection points b etween two circles and returns a list of
    (intersection_point, pair) so we know which beacons produced it."""
    d = np.linalg.norm(p2 - p1)
    if d > r1 + r2 or d < abs(r1 - r2):
        return []

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2))
    p_mid = p1 + a * (p2 - p1) / d
    offset = h * np.array([-(p2[1] - p1[1]) / d, (p2[0] - p1[0]) / d])

    return [
        (p_mid + offset, pair),
        (p_mid - offset, pair)
    ]

def find_best_intersections(beacons, distances):
    """
    Finds the best intersection points for each beacon triplet, enforcing that
    the chosen intersections come from distinct beacon pairs. If three distinct
    pairs can't be found, it falls back to choosing two intersections from
    different pairs and using their midpoint.
    """
    all_intersections = []  # For plotting or debugging
    best_beacons = None
    best_intersections = None
    final_position = None
    min_avg_distance = float('inf')

    # Iterate over all possible sets of 3 beacons
    for beacon_indices in itertools.combinations(range(len(beacons)), 3):
        i, j, k = beacon_indices
        subset_beacons = beacons[[i, j, k]]
        subset_distances = distances[[i, j, k]]

        # Produce all intersection points with pair info
        # pairs in local subset: (0,1), (0,2), (1,2)
        # map them back to the global indices (i,j,k)
        pairwise = []
        local_pairs = [(0, 1), (0, 2), (1, 2)]
        global_pairs = [(i, j), (i, k), (j, k)]

        for (lp, gp) in zip(local_pairs, global_pairs):
            a, b = lp
            ga, gb = gp  # actual beacon indices in the full array
            p1, r1 = subset_beacons[a], subset_distances[a]
            p2, r2 = subset_beacons[b], subset_distances[b]
            # circle_intersections_with_pair returns [(point, (ga, gb)), (point, (ga, gb))]
            intersections = circle_intersections_with_pair(p1, r1, p2, r2, (ga, gb))
            pairwise.extend(intersections)

        # If fewer than 2 intersection points, skip
        if len(pairwise) < 2:
            continue

        # For plotting, add just the points (without pair info) to all_intersections
        for pt, _ in pairwise:
            all_intersections.append(pt)

        # Convert to arrays for easier manipulation
        intersection_points = np.array([pt for pt, _ in pairwise])
        intersection_pairs  = [pair for _, pair in pairwise]

        # 1) Try to find three intersections from distinct pairs
        if len(intersection_points) >= 3:
            best_triplet = None
            min_distance_sum = float('inf')

            for comb in itertools.combinations(range(len(intersection_points)), 3):
                idx1, idx2, idx3 = comb
                pts = intersection_points[[idx1, idx2, idx3]]
                prs = [intersection_pairs[idx1], intersection_pairs[idx2], intersection_pairs[idx3]]

                # Distinct pair check
                if len(set(prs)) < 3:
                    # They share at least one pair => skip
                    continue

                centroid = np.mean(pts, axis=0)
                distance_sum = np.sum(np.linalg.norm(pts - centroid, axis=1))

                if distance_sum < min_distance_sum:
                    min_distance_sum = distance_sum
                    best_triplet = pts

            # If we found a valid triplet of distinct pairs, evaluate
            if best_triplet is not None:
                centroid = np.mean(best_triplet, axis=0)
                avg_distance = np.mean(np.linalg.norm(best_triplet - centroid, axis=1))
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    best_beacons = beacon_indices
                    best_intersections = best_triplet
                    final_position = centroid
                # Move on to next beacon triplet (since 3-distinct is top priority)
                continue

        # 2) If we can't find 3-distinct-pair intersections, look for 2-distinct-pair pairs
        # We'll pick the pair of points with the smallest distance
        best_pair = None
        min_dist = float('inf')
        for comb in itertools.combinations(range(len(intersection_points)), 2):
            idx1, idx2 = comb
            pt1, pt2 = intersection_points[idx1], intersection_points[idx2]
            pair1, pair2 = intersection_pairs[idx1], intersection_pairs[idx2]

            if pair1 == pair2:
                # Same pair => skip
                continue

            dist = np.linalg.norm(pt1 - pt2)
            if dist < min_dist:
                min_dist = dist
                best_pair = (pt1, pt2)

        # If we found a valid 2-distinct-pair pair, evaluate
        if best_pair is not None:
            midpoint = (best_pair[0] + best_pair[1]) / 2.0
            avg_distance = min_dist / 2.0  # distance from midpoint to each point
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_beacons = beacon_indices
                best_intersections = np.array(best_pair)  # store the 2 points
                final_position = midpoint

    return final_position, best_beacons, all_intersections, best_intersections


def find_best_intersections2(beacons, distances):
    """Detects the densest cluster and finds its centroid. 
    Returns the centroid and the associated confidence interval."""
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

    # Convert to NumPy array
    all_intersections = np.array(all_intersections) # Apply DBSCAN
    eps = 0.2  # Maximum distance between points in a cluster
    eps_max = 60
    min_samples = 2  # Minimum points to form a cluster

    x_min, y_min = np.min(beacons, axis=0)
    x_max, y_max = np.max(beacons, axis=0)

    while np.all(estimated_position == np.array([None, None])):
        #print("scanning for clusters...", eps)
        if eps > eps_max:
            estimated_position = np.mean(all_intersections, axis=0)
            best_cluster_points = all_intersections
            #print("exceeded eps limit")
            break
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(all_intersections)
        labels = np.array(db.labels_)
        valid_clusters = labels[labels != -1]

        if len(valid_clusters) > 0:
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            best_cluster_label = unique_labels[np.argmax(counts)]
            best_cluster_points = all_intersections[labels == best_cluster_label]

            # Compute final estimated position (mean of best cluster)
            estimated_position = best_cluster_points.mean(axis=0)

            if estimated_position[0] < x_min-2 or estimated_position[0] > x_max+2:
                #print("position out of bounds, finding a new one")
                estimated_position = np.array([None, None])
                labels[labels == best_cluster_label] = -1
                valid_clusters = labels[labels != -1]
                if eps<2:
                    eps+=0.2
                else:
                    eps +=1

            elif estimated_position[1] < y_min-2 or estimated_position[1] > y_max+2:
                #print("position out of bounds, finding a new one")
                estimated_position = np.array([None, None])
                labels[labels == best_cluster_label] = -1
                valid_clusters = labels[labels != -1]
                if eps<2:
                    eps+=0.2
                else:
                    eps +=1
        else:
            if eps<2:
                eps+=0.2
            else:
                eps +=1
            #print("no clusters found, retrying")


    residuals = []
    for i, beacon in enumerate(beacons):
        #print("pos: ", estimated_position)
        r_i = distances[i]
        dist_est = np.linalg.norm(estimated_position - beacon)
        residuals.append((dist_est - r_i)**2)
        #print("res: (dist_est - r_i)**2")

    SSE = sum(residuals)
    #print(SSE)
    alpha = 0.5
    confidence = 1.0 / (1.0 + alpha* np.sqrt(SSE))
    #print("confidence: ", confidence)

    #cluster_size = len(best_cluster_points)
    #spread = np.mean(np.linalg.norm(best_cluster_points - estimated_position, axis=1))
    #alpha = 10e5  # Tune this parameter; higher alpha means more sensitivity.
    #confidence = (cluster_size / (cluster_size + 2)) * np.exp(-alpha * spread)
    #confidence = (cluster_size / (cluster_size + 2)) * (1.0 / (1.0 + 10 * spread))
    #confidence = 1.0 / (spread*0.5 + 1)
    #if we use this - no current best_beacons data
    #so change the return or implement a way of finding it

    return estimated_position, confidence, all_intersections, best_cluster_points

def plot_results(beacons, distances, all_intersections, best_intersections, final_position):
    """Plots beacons, circles, all intersections, selected intersections, and estimated position."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw circles for beacons
    for i, (beacon, radius) in enumerate(zip(beacons, distances)):
        circle = plt.Circle(beacon, radius, color=f"C{i}", alpha=0.3, fill=False)
        ax.add_patch(circle)
        ax.scatter(*beacon, color=f"C{i}", label=f"Beacon {i+1}")
        ax.text(beacon[0], beacon[1], f"B{i+1}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # Convert lists to numpy arrays for plotting
    all_intersections = np.array(all_intersections)
    best_intersections = np.array(best_intersections)

    # Plot all intersections in black
    if all_intersections.size > 0:
        ax.scatter(all_intersections[:, 0], all_intersections[:, 1], color="black", marker="x", label="All intersections")

    # Plot best intersections in green
    if best_intersections is not None and best_intersections.size > 0:
        ax.scatter(best_intersections[:, 0], best_intersections[:, 1], color="green", marker="x", label="Best intersections")

    # Plot final estimated position in red
    ax.scatter(*final_position, color="red", marker="*", s=200, label="Final estimated position")

    ax.set_xlim(beacons[:, 0].min() - 2, beacons[:, 0].max() + 2)
    ax.set_ylim(beacons[:, 1].min() - 2, beacons[:, 1].max() + 2)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Trilateration with Best Intersection Points")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()


def trilaterate(df, beacon_positions):
    """Computes player positions based on beacon distances and adds them to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing distance measurements from beacons.
        beacon_positions (np.ndarray): Array of shape (N, 2) representing beacon coordinates.

    Returns:
        pd.DataFrame: A copy of df with added 'pos_x' and 'pos_y' columns.
    """

    # Identify distance columns (starting with 'b')
    distance_columns = [col for col in df.columns if col.startswith('b')]
    
    # Create a copy of the DataFrame
    df_result = df.copy()

    # Initialize position lists
    pos_x, pos_y = [], []
    confidence_list = []

    # Dictionary to count how often each beacon is used
    beacon_usage_count = defaultdict(int)

    for index, row in df.iterrows():
        distances = np.array(row[distance_columns])

        try:
            # Compute best trilateration
            final_position, confidence, all_intersections, best_intersections = find_best_intersections2(beacon_positions, distances)
            pos_x.append(final_position[0])
            pos_y.append(final_position[1])
            confidence_list.append(confidence)
            
            #plot_results(beacon_positions, distances, all_intersections, best_intersections, final_position)

        except ValueError:
            # If trilateration fails, append NaN
            print("trilateration threw an error")
            pos_x.append(np.nan)
            pos_y.append(np.nan)
            confidence_list.append(np.nan)


    # Add computed positions to the DataFrame
    df_result['pos_x'] = pos_x
    df_result['pos_y'] = pos_y
    df_result['confidence'] = confidence_list

    # print("Beacon Usage Count:")
    # for beacon_idx, count in sorted(beacon_usage_count.items()):
    #     print(f"Beacon {beacon_idx}: {count} times")

    return df_result