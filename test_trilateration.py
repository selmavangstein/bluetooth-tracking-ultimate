import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from final_trilateration import find_best_intersections
from final_trilateration import find_best_intersections2


def plot_results(beacons, distances, all_intersections, best_intersections, final_position, timestamp):
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
    if best_intersections != None and best_intersections.size > 0:
        ax.scatter(best_intersections[:, 0], best_intersections[:, 1], color="green", marker="x", label="Best intersections")

    # Plot final estimated position in red
    ax.scatter(*final_position, color="red", marker="*", s=200, label="Final estimated position")

    #ax.set_xlim(beacons[:, 0].min() - 2, beacons[:, 0].max() + 2)
    #ax.set_ylim(beacons[:, 1].min() - 2, beacons[:, 1].max() + 2)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Trilateration with Best Intersection Points at {timestamp.time()}")
    ax.legend()
    ax.set_aspect("equal")
    plt.show()


#testdata = pd.read_csv("processedtest6.csv")
testdata = pd.read_csv("data/GT-obstacletest-UWB-feb5.csv")
row = testdata.iloc[2]  # Select a specific row for testing
timestamp = pd.to_datetime(row['timestamp'])
distances = np.array(row[[col for col in testdata.columns if col.startswith('b')]])
beacons = np.array([[0, 0], [12, 0], [0, 18], [12, 18]])  
#beacons = np.array([[0, 0], [15, 0], [0, 20], [15, 20]])

# Step 1: Find best intersection points and final position
final_position, confidence, all_intersections, best_intersections = find_best_intersections2(beacons, distances)
#print("# of all intersections: ", (all_intersections))
#print("# of best intersections: ", (best_intersections))

# Step 2: Plot the results
plot_results(beacons, distances, all_intersections, best_intersections, final_position, timestamp)
