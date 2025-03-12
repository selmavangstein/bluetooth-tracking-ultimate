import numpy as np

"""File to process acceleration data. 
Finding the acceleration vector relative to the beacons is hard,
and depends on the current position coordinate, so we dropped this idea.
We now only work with acceleration magnitudes.
We still recommend working with the acceleration vector relative to north, and use this in a
2D Kalman Filter that takes both position and acceleration measurements (not currently implemented)"""

def find_beacon_acceleration(acceleration_data, compass_angle):
    '''
    DOES NOT CURRENTLY WORK. NEEDS A WHOLE LOT OF WORK.
    Finds the components of the acceleration in our coordinate system.
    We will always have ax=-9.81, and use this with the compass angle to find ay,az
    Beacon angles cannot be hard coded. Create a fcn that takes trilaterated data to estimate them.
    '''

    ax_i = acceleration_data[0]
    ay_i = acceleration_data[1]
    az_i = acceleration_data[2]
    alpha = compass_angle

    #these can't be hardcoded, we need to find them based on position...
    #so this whole pipeline cannot be separate from the position data, they need to both be considered.
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


def find_acceleration_magnitude(df):
    """Find the acceleration magnitude at each time stamp.
    Args:
        df: contains acceleration data. Must contain the headers 'xa', 'ya', 'za'
    Returns:
        df: a copy of df with a new column 'ta' giving the total acceleration magnitude
    """
    df = df.copy()
    xa = df['xa']
    ya = df['ya']
    za = df['za']
    #note that we here subtract the acceleration due to gravity. This might not always be pointing in the y-direction.
    #If the down-direction is roughly constant throughout the test, this approach works, but if down is x,y, or z might vary
    #between setups
    #If the down-direction changes (like if the player dives or decides to do a backflip) this approximation will not be accurate
    acceleration_magnitudes = np.sqrt(xa**2 + (ya-9.8)**2 + za**2)
    df['ta'] = acceleration_magnitudes
    return df