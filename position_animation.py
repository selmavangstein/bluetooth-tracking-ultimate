

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay 
from trilateration import trilaterate
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
import json

# setup
with open('first_test.log') as f:
    lines = f.readlines()

smoothing = True

""" New format
[10:43:26.098] {"beacon2":[12.90,86,-46],"beacon3":[0.45,3,-42]}
[10:43:26.098] {"beacon2":conERR,"beacon3":[0.60,4,-43]}
[10:43:26.098] {"beacon2":[12.00,80,-45],[0.00,0,-45]}
[10:43:26.098] {"beacon2":[12.00,80,-45],"beacon3":[41.10,274,-46]}
[10:43:26.099] {"beacon2":[12.30,82,-45],"beacon3":[1.20,8,-45]}
[10:43:26.099] {"beacon2":[12.15,81,-46],"beacon3":[0.60,4,-45]}
[10:43:26.233] {"beacon2":[12.60,84,-46],"beacon3":[0.45,3,-45]}
[10:43:26.478] {"beacon2":[12.15,81,-46],"beacon3":[0.30,2,-45]}"""


b1 = [0]
b2 = [0]
b3 = [0]
ts = [0]

# new parsing method
lines = [l for l in lines if '\n' != l]
# print(lines)
i=0
for line in lines:
    data = str(line.split(' ')[1].replace('\n', ''))

    print(data)
    if (data[0] != '{' and data[-1] != '}') or "wifi:" in data:
        continue

    #covnert to dict
    try:
        data = json.loads(data)
    except:
        continue

    try:
        b1d = data['beacon1'][0]
    except:
        b1d = b1[i]

    try:
        b2d = data['beacon2'][0]
    except:
        b2d = b2[i]

    try:
        b3d = data['beacon3'][0]
    except:
        b3d = b3[i]

    if b1d == 'c' or b2d == 'c' or b3d == 'c':
        continue

    b1.append(b1d)
    b2.append(b2d)
    b3.append(b3d)
    ts.append(line.split(' ')[0])

    # if type(b1) != int:
    #     b1 = b1[i]
    # if type(b2) != int:
    #     b2 = b2[i]
    # if type(b3) != int: 
    #     b3 = b3[i]

    i += 1


"""
old log file parsing
i = 0
for line in lines:
    if 'beacon' in line:
        
        data = str(lines[i+1]).split(' ')[1].split(',')[0] # get distance value

        print(lines[i+1])
        
        if 'beacon' in data:
            continue
        else:
            data = float(data)

        if 'beacon1' in line:
            b1.append(data)
            b2.append(b2[-1])
            b3.append(b3[-1])
        elif 'beacon2' in line:
            b2.append(data)
            b1.append(b1[-1])
            b3.append(b3[-1])
        elif 'beacon3' in line:
            b3.append(data)
            b1.append(b1[-1])
            b2.append(b2[-1])
    
    i+=1"""


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
        [15, 26],    # Beacon 1 at origin
        [0, 0],   # Beacon 2 at (10,0)
        [30, 0]    # Beacon 3 at (0,10)
    ])

    # Define distances from each beacon to the target
    distances = np.array([b1[i], b2[i], b3[i]])

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    # plot the beacons
    plt.plot(15, 26, color='green', marker='o', markersize=10)
    plt.plot(0, 0, color='green', marker='o', markersize=10)
    plt.plot(30, 0, color='green', marker='o', markersize=10)
    plt.plot(14, 11, color='orange', marker='o', markersize=10) # middle point

    try:
        position = trilaterate(beacons, distances)
        print(f"Calculated position: ({position[0]:.2f}, {position[1]:.2f})")
        # plt.plot(position[0], position[1], 'ro', alpha=a)
        plt.plot(position[0], position[1], 'ro', alpha=a, color=plt.cm.viridis(a))
        plt.pause(0.01)
    except ValueError as e:
        print(f"Error: {e}")

    a += step

    

plt.show()


