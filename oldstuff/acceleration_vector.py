import numpy as np

def find_acceleration(acceleration_data, compass_angle):
    '''
    Finds the components of the acceleration in our coordinate system.
    We will always have ax=-9.81, and use this with the compass angle to find ay,az
    '''

    ax_i = acceleration_data[0]
    ay_i = acceleration_data[1]
    az_i = acceleration_data[2]
    alpha = compass_angle

    #beacon angle relative to north
    b1 = 14
    b2 = 35
    b3 = 43
    b4 = 67

    beacon_angles = [b1, b2, b3, b4]
    acceleration_beacon_components = []

    absolute_acceleration = np.sqrt(ax_i**2 + ay_i**2 + az_i**2)
    ax = -9.81
    for angle in beacon_angles:
        a_b = np.sqrt(absolute_acceleration**2 - ax**2) * np.cos(alpha + angle)
        acceleration_beacon_components.append(a_b)

    #ay = np.sqrt(absolute_acceleration**2 - ax**2) * np.sin(alpha)
    #az = np.sqrt(absolute_acceleration**2 - ax**2) * np.cos(alpha)

    return acceleration_beacon_components

'''
For pipeline creator: these need to be stored in the data csv. 
I recommend splitting them up so that we have file["acc-beacon-1"], file["acc-beacon-2"] etc for each beacon.
This is at least what I am assuming in the kalman filter code - if you do something else, update the Kalman
file accordingly please.
'''