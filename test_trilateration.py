import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from final_trilateration import find_best_intersections
from final_trilateration import find_best_intersections2


def plot_results(beacons, distances, all_intersections, best_intersections, final_position, gt, timestamp):
    """Plots beacons, circles, all intersections, selected intersections, and estimated position."""
    plt.ion()

    fig, ax = plt.subplots(figsize=(8, 8))
    #ax.set_xlim(beacons[:, 0].min() - 2, beacons[:, 0].max() + 2)
    #ax.set_ylim(beacons[:, 1].min() - 2, beacons[:, 1].max() + 2) 
    ax.set_xlim(8.5,10.5)
    ax.set_ylim(4.5,7.5)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(f"Final coordinate at t={timestamp.time()}")
    ax.legend()
    ax.set_aspect("equal")

    # Draw circles for beacons
    for i, (beacon, radius) in enumerate(zip(beacons, distances)):
        circle = plt.Circle(beacon, radius, color=f"C{i}", alpha=0.3, fill=False)
        ax.add_patch(circle)
        ax.scatter(*beacon, color=f"C{i}", label=f"Beacon {i+1}")
        ax.text(beacon[0], beacon[1], f"B{i+1}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.draw()            # Update the figure
    plt.pause(1.5)

    # Convert lists to numpy arrays for plotting
    all_intersections = np.array(all_intersections)
    best_intersections = np.array(best_intersections)

    #ax.scatter(gt['pos_x'], gt['pos_y'], marker="x", color="grey", label="ground truth")

    # Plot all intersections in black
    if all_intersections.size > 0:
        ax.scatter(all_intersections[:, 0], all_intersections[:, 1], color="black", marker="x", label="All intersections")
    plt.draw()
    plt.pause(1.5)
    # Plot best intersections in green
    #if best_intersections != None and best_intersections.size > 0:
    ax.scatter(best_intersections[:, 0], best_intersections[:, 1], color="green", marker="X", s=100, label="Best intersections")
    plt.draw()            # Update the figure
    plt.pause(1.5)

    # Plot final estimated position in red
    ax.scatter(*final_position, color="red", marker="*", s=100, label="Final estimated position")
    plt.draw()            # Update the figure
    plt.pause(1.5)

    # Loop through each beacon
    for i, beacon in enumerate(beacons):
        r_i = distances[i]
            
        # Compute the distance from the beacon to the estimated position.
        d = np.linalg.norm(final_position - beacon)
        
        # Compute the closest point on the circle:
        # This is beacon + (vector from beacon to estimated_position scaled to have length r_i)
        if d != 0:
            closest_point = beacon + (final_position - beacon) * (r_i / d)
        else:
            closest_point = beacon  # Edge case: estimated_position equals beacon
        
        # Draw a dashed line from the estimated position to the closest point on the circle.
        ax.plot([final_position[0], closest_point[0]],
                [final_position[1], closest_point[1]],
                'k--', linewidth=1)
        

    plt.ioff()
    plt.show()



#testdata = pd.read_csv("processedtest6.csv")
testdata = pd.read_csv("player_positions_Velocity Clamping5.csv")
gt_df = pd.read_csv("player_positions_Ground Truth.csv")

testdata['timestamp'] = pd.to_datetime(testdata['timestamp'])
gt_df['timestamp'] = pd.to_datetime(gt_df['timestamp'])

row = testdata.iloc[567]  #Select a specific row for testing

timestamp = row['timestamp']
gt_df['time_diff'] = (gt_df['timestamp'] - timestamp).abs()
gt_row = gt_df.loc[gt_df['time_diff'].idxmin()]
print(gt_row)


distances = np.array(row[[col for col in testdata.columns if col.startswith('b')]])
beacons = np.array([[0, 0], [12, 0], [0, 18], [12, 18]])  
#beacons = np.array([[0, 0], [15, 0], [0, 20], [15, 20]])

# Step 1: Find best intersection points and final position
final_position, confidence, all_intersections, best_intersections = find_best_intersections2(beacons, distances)

# Step 2: Plot the results
plot_results(beacons, distances, all_intersections, best_intersections, final_position, gt_row, timestamp)
