

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay 
from trilateration import trilaterate_one
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import json

# setup
data = pd.read_csv("/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/4beaconv1.csv")
smoothing = True

b1 = [0]
b2 = [0]
b3 = [0]
b4 = [0]
ts = [0]

b1 = data['b1d'].tolist()
b2 = data['b2d'].tolist()
b3 = data['b3d'].tolist()
b4 = data['b4d'].tolist()
ts = data['timestamp'].tolist()


# remove the first element
b1.pop(0)
b2.pop(0)
b3.pop(0)

print(b1)
print(b2)
print(b3)

if smoothing:
    # smooth the data
    x = []
    for b in [b1, b2, b3]:
        param_series = pd.Series(b)
        emaz = param_series.ewm(span=25, adjust=False).mean().dropna().tolist()
        x.append(emaz)

    b1 = x[0]
    b2 = x[1]
    b3 = x[2]

a = 0.1
step = 0.9 / len(b1)
for i in range(len(b1)):
    # Define three beacon positions
    beacons = np.array([
        [2000, 0],    # Beacon 1 at origin
        [0, 0],   # Beacon 2 at (10,0)
        [0, 4000]    # Beacon 3 at (0,10)
    ])

    # Define distances from each beacon to the target
    distances = np.array([b1[i], b2[i], b3[i]])

    # plot the beacons
    plt.plot(beacons[0][0], beacons[0][1], color='green', marker='o', markersize=10)
    plt.plot(beacons[1][0], beacons[1][1], color='green', marker='o', markersize=10)
    plt.plot(beacons[2][0], beacons[2][1], color='green', marker='o', markersize=10)

    try:
        position = trilaterate_one(beacons, distances)
        print(f"Calculated position: ({position[0]:.2f}, {position[1]:.2f})")
        # plt.plot(position[0], position[1], 'ro', alpha=a)
        plt.plot(position[0], position[1], 'ro', alpha=a, color=plt.cm.viridis(a))
        plt.pause(0.01)
    except ValueError as e:
        print(f"Error: {e}")

    a += step

    

plt.show()


