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
    absolute_acceleration = np.sqrt(ax_i**2 + ay_i**2 + az_i**2)
    ax = -9.81
    ay = np.sqrt(absolute_acceleration**2 - ax**2) * np.sin(alpha)
    az = np.sqrt(absolute_acceleration**2 - ax**2) * np.cos(alpha)
