import numpy as np
import itertools
from collections import defaultdict


def circle_intersections(p1, r1, p2, r2):
    """Finds intersection points between two circles."""
    d = np.linalg.norm(p2 - p1)
    
    if d > r1 + r2 or d < abs(r1 - r2):  # No intersection
        return []

    # Compute the intersection points
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2))

    p_mid = p1 + a * (p2 - p1) / d
    offset = h * np.array([-(p2[1] - p1[1]) / d, (p2[0] - p1[0]) / d])

    return [p_mid + offset, p_mid - offset]

from sklearn.cluster import KMeans

def find_best_trilateration(beacons, distances):
    """Finds the best trilateration estimate using clustering and MSE minimization, with additional cluster refinement."""
    all_intersections = []  # Store all intersections found
    best_beacons = None
    best_intersections = None
    min_mse = float('inf')
    best_cluster_center = None

    # Iterate over all possible sets of 3 beacons
    for beacon_indices in itertools.combinations(range(len(beacons)), 3):
        i, j, k = beacon_indices
        subset_beacons = beacons[[i, j, k]]
        subset_distances = distances[[i, j, k]]

        # Compute all pairwise intersections
        points = (
            circle_intersections(subset_beacons[0], subset_distances[0], subset_beacons[1], subset_distances[1]) +
            circle_intersections(subset_beacons[0], subset_distances[0], subset_beacons[2], subset_distances[2]) +
            circle_intersections(subset_beacons[1], subset_distances[1], subset_beacons[2], subset_distances[2])
        )

        if len(points) < 3:
            continue  # Skip if not enough intersections

        all_intersections.extend(points)  # Save all intersections

        points = np.array(points)

        # Perform clustering to find the densest region
        kmeans = KMeans(n_clusters=1, n_init=10, random_state=42).fit(points)
        cluster_center = kmeans.cluster_centers_[0]
        mse = np.mean(np.linalg.norm(points - cluster_center, axis=1) ** 2)

        # Choose the best clustering result based on lowest MSE
        if mse < min_mse:
            min_mse = mse
            best_beacons = beacon_indices
            best_intersections = points
            best_cluster_center = cluster_center

    if best_beacons is None:
        raise ValueError("No valid beacon triplet found for trilateration")

    # Refine the best cluster using DBSCAN
    #refined_position = find_best_cluster(best_intersections)

    return best_cluster_center, best_beacons, all_intersections, best_intersections

import numpy as np
from sklearn.cluster import DBSCAN

def find_best_cluster(intersections, eps=2, min_samples=2):
    """
    Cluster intersection points and select the best cluster based on average distance to centroid.
    """
    if len(intersections) < min_samples:
        return np.mean(intersections, axis=0)  # If too few points, just return their mean
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(intersections)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    best_cluster = None
    best_centroid = None
    min_avg_distance = float('inf')
    
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points
        
        cluster_points = intersections[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        avg_distance = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
        
        if avg_distance < min_avg_distance:
            min_avg_distance = avg_distance
            best_cluster = cluster_points
            best_centroid = centroid
    
    return best_centroid


def trilaterate_one(beacons, distances):
    P1, P2, P3 = beacons
    r1, r2, r3 = distances
    
    P21 = P2 - P1
    P31 = P3 - P1
    
    A = 2 * np.array([
        [P21[0], P21[1]],
        [P31[0], P31[1]]
    ])
    
    b = np.array([
        r1**2 - r2**2 - np.dot(P1, P1) + np.dot(P2, P2),
        r1**2 - r3**2 - np.dot(P1, P1) + np.dot(P3, P3)
    ])
    
    try:
        position = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    return position

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

    # Dictionary to count how often each beacon is used
    beacon_usage_count = defaultdict(int)

    for _, row in df.iterrows():
        distances = np.array(row[distance_columns])

        try:
            # Compute best trilateration
            final_position, best_beacons, _, _ = find_best_trilateration(beacon_positions, distances)
            pos_x.append(final_position[0])
            pos_y.append(final_position[1])

            for beacon_idx in best_beacons:
                beacon_usage_count[beacon_idx] += 1

        except ValueError:
            # If trilateration fails, append NaN
            pos_x.append(np.nan)
            pos_y.append(np.nan)

    # Add computed positions to the DataFrame
    df_result['pos_x'] = pos_x
    df_result['pos_y'] = pos_y

    print("Beacon Usage Count:")
    for beacon_idx, count in sorted(beacon_usage_count.items()):
        print(f"Beacon {beacon_idx}: {count} times")

    return df_result


#OLD TEST FILE

def plot_intersections(beacons, distances, all_intersections, best_intersections, final_position):
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
    if best_intersections.size > 0:
        ax.scatter(best_intersections[:, 0], best_intersections[:, 1], color="green", marker="x", label="Best intersections")

    # Plot final estimated position in red
    ax.scatter(*final_position, color="red", marker="*", s=200, label="Final estimated position")

    ax.set_xlim(beacons[:, 0].min() - 2, beacons[:, 0].max() + 2)
    ax.set_ylim(beacons[:, 1].min() - 2, beacons[:, 1].max() + 2)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Best Trilateration Based on Circle Intersections")
    ax.legend()
    ax.set_aspect("equal")
    
    plt.show()

def plot_trilateration(beacons, radii, all_intersections, best_cluster, final_position):
    fig, ax = plt.subplots()

    # Plot all beacons
    colors = ['blue', 'orange', 'green', 'red']
    for i, (beacon, color) in enumerate(zip(beacons, colors)):
        ax.scatter(*beacon, color=color, label=f"Beacon {i+1}")
        ax.text(beacon[0], beacon[1], f"B{i+1}", fontsize=12, verticalalignment='bottom')

    # Plot all circles
    for beacon, radius, color in zip(beacons, radii, colors):
        circle = plt.Circle(beacon, radius, color=color, fill=False, alpha=0.3)
        ax.add_patch(circle)

    # Plot all intersections (black)
    if all_intersections:
        all_intersections = np.array(all_intersections)
        ax.scatter(all_intersections[:, 0], all_intersections[:, 1], color='black', label="All intersections", marker='x')

    # Plot best cluster intersections (green)
    if best_cluster:
        best_cluster = np.array(best_cluster)
        ax.scatter(best_cluster[:, 0], best_cluster[:, 1], color='green', label="Best intersections", marker='x')

    # Plot final estimated position (red star)
    ax.scatter(*final_position, color='red', marker='*', s=200, label="Final estimated position")

    ax.legend()
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Best Trilateration Based on Circle Intersections")

    plt.show()



# Load dataset
testdata = pd.read_csv("processedtest6.csv")
row = testdata.iloc[150]  # Select a specific row for testing
distances = np.array(row[[col for col in testdata.columns if col.startswith('b')]])
beacons = np.array([[0, 0], [12, 0], [0, 16], [12, 16]])  # Hardcoded beacon locations

# Step 1: Compute all intersections using trilateration
final_position, best_beacons, all_intersections, best_intersections = find_best_trilateration(beacons, distances)

""" # Step 2: Use clustering to refine the best cluster of intersections
best_intersections = np.array(all_intersections)  # Convert list to array
if len(best_intersections) > 0:
    refined_position = find_best_cluster(best_intersections)
else:
    refined_position = final_position  # Fallback if no intersections """

refined_position = final_position

# Step 3: Plot the results
plot_intersections(beacons, distances, all_intersections, best_intersections, refined_position)