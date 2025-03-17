
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def twoD_correction(locations, timestamps, acc, ema_window=100):
    """
    Attempts to correct the 2d trilateration data if there are big jumps
    Almost like another kalman filter (but 2d)

    takes in a list like : [[x,y], [x1,y1], ... [xn,yn]]
    """

    # Store the corrected data
    corrections = [locations[0]]

    # calculate an ema of the data
    # convert to a pandas dataframe
    locations_df = pd.DataFrame(locations, columns=['x', 'y'])

    # comvert timestamps to numbers
    timestamps = [pd.Timestamp(ts).timestamp() for ts in timestamps]

    corrected = 0
    total = 0

    # Loop through the data
    for i in range(1, len(locations)):
        # if the distance between two points is greater than 10m
        # get the time difference between the points
        time_diff = (timestamps[i] - timestamps[i-1])
        
        # time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
        distance_diff = np.linalg.norm(locations[i] - corrections[i-1])

        prev_distance_diff = np.linalg.norm(locations[i-1] - corrections[i-1]) * time_diff

        scaler = time_diff * 10

        if distance_diff > scaler:
            # update ema
            correction_df = pd.DataFrame(corrections, columns=['x', 'y'])
            ema = correction_df.ewm(span=ema_window, adjust=False).mean()
            ema = ema[['x', 'y']].values

            # calculate the 10m circle around the previous point
            circle = np.array([corrections[i-1] + np.array([scaler * np.cos(theta), scaler * np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, 100)]) # mostly used for plotting
            correct_circle = np.array([corrections[i-1] + np.array([prev_distance_diff * np.cos(theta), prev_distance_diff * np.sin(theta)]) for theta in np.linspace(0, 2 * np.pi, 100)]) # calc new distanace
            closest_point = min(correct_circle, key=lambda point: np.linalg.norm(point - ema[i-1]))

            corrections.append(closest_point)
            """quickly plot the correction for testing"""
            # plt.plot(corrections[i-1][0], corrections[i-1][1], 'yo', label='Prev Point')
            # plt.plot(locations[i][0], locations[i][1], 'bo', label='Current Point')
            # plt.plot(circle[:, 0], circle[:, 1], label='Impossible Circle')
            # plt.plot(ema[i-1][0], ema[i-1][1], 'ro', label='EMA Point')
            # plt.plot(closest_point[0], closest_point[1], 'go', label='Corrected Point (what is added)')
            # plt.legend()
            # plt.show()
            # plt.close()
            corrected += 1
            total += 1

        else:
            total += 1
            corrections.append(locations[i])

    print(f"Corrected {corrected} out of {total} points")
    return np.array(corrections)

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
