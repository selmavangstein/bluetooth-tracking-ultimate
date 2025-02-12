from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
from residualcheck import residualcheck
import matplotlib.pyplot as plt

from acceleration_vector import find_acceleration_magnitude

def pos_vel_filter(x, P, R, Q=0., dt=1.):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.array([x[0], x[1], x[2]]) #initial state
    kf.F = np.array([[1., dt, 1/2*dt**2],
                     [0.,  1., dt],
                     [0., 0., 1]])  #transition matrix
    kf.H = np.array([[1., 0, 0.]])    #Measurement function
    if np.isscalar(R):
        kf.R *= R                 # covariance matrix 
    else:
        kf.R[:] = R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 #initial covariance matrix 
    else:
        kf.P[:] = P
    if np.isscalar(Q):          #process noise model
        kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf

def kalman_filter(zs, ta, times, smoothing=True):
    '''Takes measurements and timestamps (must be on datetime format). Returns filtered data'''

    times = pd.to_datetime(times)
    time_differences = times.diff()
    average_difference = time_differences.dt.total_seconds().mean()

    dt = average_difference
    x = np.array([zs[0], 0, 0]) #initial state.
    P = np.diag([.2, 9., 5.]) #initial state uncertainty 
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=100000*dt) #process noise.
    R = np.array([[0.2]]) #measurement covariance matrix /sensor variance.

    f = pos_vel_filter(x, P, R, Q, dt)
    s = Saver(f)
    for i in range(0, len(zs)):
        #can change f.F here to reflect dt fluctuations.
        #generally worth it with fluctuations more than 10% from the mean
        #should then also update f.Q.
        f.predict()

        estimated_acc = abs(f.x[2])  # Assuming acceleration is the 3rd state variable (index 2)
        if estimated_acc > ta[i]:  
            f.x[2] = np.sign(f.x[2]) * ta[i]  # Scale down but keep direction

        #uses the magnitude of the acceleration along with estimated velcoity as a max limit for position change
        if i>=1:
            max_pos_change = abs(f.x[1]) * dt + 0.5 *  ta[i]* dt**2
            predicted_change = abs(f.x[0] - zs[i-1])
            if predicted_change > max_pos_change:
                f.x[0] = zs[i-1] + np.sign(f.x[1]) * max_pos_change

        f.update(zs[i])
        s.save()

    s.to_array()
    xs = s.x
    covs = s.P
    smooth_xs = None
    if smoothing:
        smooth_xs, smooth_cov, _, _ = f.rts_smoother(xs, covs)

    residualcheck(s, R)
    #this returns a saver object with all the information about the filter
    return s, smooth_xs

def pipelineKalman(df):
    df = df.copy()  
    df = find_acceleration_magnitude(df) # adds acceleration vectors to the df

    results = {}
    for column in df.columns:
        if column.startswith('b'):
            zs = df[column].values
            s, smooth_xs = kalman_filter(zs, df['ta'].values, df['timestamp'], smoothing=True)

            xs = s.x
            results[column] = xs[:, 0]  # store the position estimates


    # Add the results to the dataframe, replacing the original data
    for column, values in results.items():
        df[column] = values

    return df

