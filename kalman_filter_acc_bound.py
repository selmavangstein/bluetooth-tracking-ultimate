from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
#from kf_book.mkf_internal import plot_track
# from kalman_plotting import plot_results
import matplotlib.pyplot as plt

from acceleration_vector import find_acceleration_magnitude

def pos_vel_filter(x, P, R, Q=0., dt=1.):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]]) # location and velocity. 
    kf.F = np.array([[1., dt],
                     [0.,  1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])    # Measurement function
    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix 
    else:
        kf.P[:] = P               # [:] makes deep copy if matrix is already given
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf

def kalman_filter(zs, ta, times, smoothing=True):
    '''Takes measurements and timestamps (must be on datetime format). Returns filtered data'''

    times = pd.to_datetime(times)
    time_differences = times.diff()
    average_difference = time_differences.dt.total_seconds().mean()

    dt = average_difference
    x = np.array([zs[0], 0]) #initial state.
    P = np.diag([0.2, 0.2]) #initial state uncertainty 
    #(velocity p should be max 9. If we make sure they are held still for a few seconds at initialization, we can lower it to like 1)
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=20*dt) #process noise. Used change high bound for max accel. times dt. Might lower because model isn't great, so must trust measurements more.
    R = np.array([[0.2]]) #measurement covariance matrix /sensor variance. Use variance testing to decide this

    f = pos_vel_filter(x, P, R, Q, dt)
    s = Saver(f)
    for i in range(0, len(zs)):
        #can change f.F here to reflect dt fluctuations.
        #generally worth it with fluctuations more than 10% from the mean
        #should then also update f.Q.
        f.predict()

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

    

    #this returns a saver object with all the information about the filter
    return s, smooth_xs

def pipelineKalman(df):
    df = df.copy()  
    df = find_acceleration_magnitude(df) # adds acceleration vectors to the df

    results = {}
    for column in df.columns:
        if column.startswith('b'):
            zs = df[column].values
            xs, smooth_xs = kalman_filter(zs, df['ta'].values, df['timestamp'], smoothing=True)
            results[column] = xs[:, 0]  # store the position estimates

    # Add the results to the dataframe, replacing the original data
    for column, values in results.items():
        df[column] = values

    return df

