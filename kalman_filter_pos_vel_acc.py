from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
#from kf_book.mkf_internal import plot_track
from kalman_plotting import plot_results
import matplotlib.pyplot as plt


def pos_vel_acc_filter(x, P, R, Q=0., dt=1.):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.x = np.array([x[0], x[1], x[2]]) # initial location, velocity, acceleration 
    kf.F = np.array([[1., dt, 1/2*dt**2],
                     [0.,  1., dt],
                     [0., 0., 1]])  # state transition matrix
    kf.H = np.array([[1., 0, 0.], 
                     [0., 0, 1.]])    # Measurement function
    if np.isscalar(R):
        kf.R *= R                 # covariance matrix 
    else:
        kf.R[:] = R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix 
    else:
        kf.P[:] = P               # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf

def kalman_filter(pos_data, acc_data, times, smoothing=True):
    time_differences = times.diff()
    average_difference = time_differences.dt.total_seconds().mean()

    dt = average_difference

    x = np.array([10.0, 0, 0]) #initial state. Use first measurement.
    P = np.diag([70, 16, 4]) #initial state uncertainty 
    R = np.array([[3., 0.],
                [0., 5.]]) #measurement covariance matrix /sensor variance. Current value is a guess.
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=2.35) #process noise - var is guess, adapt later

    """df_data = pd.read_csv("bluetooth-tracking-ultimate\makecharts\walking test to 60_cleaned.csv")
    pos_data = df_data["distance"]
    acc_data = df_data["acc_b1"] """

    zs = list(zip(pos_data, acc_data))

    f = pos_vel_acc_filter(x, P, R, Q, dt)
    s = Saver(f)
    xs, covs, _, _ = f.batch_filter(zs, saver=s)
    smooth_xs = None
    if smoothing:
        smooth_xs, smooth_cov, _, _ = f.rts_smoother(xs, covs)
    #plot xs
    s.to_array()
    #plot_results(s.x[:, 0], s.z, s.P) #z is measurement, x is filtered value, P is variance
    return s, smooth_xs

    '''
    FOR PIPELINE CREATOR
    Here I am just plotting the results with some stolen code - feel free to look at it as an example of how to handle the
    data. Not sure what is best for you that I return here, depending on what we want to vizualise.
    The saver object contains all the relevant data, so might return that when we make a function out of this.
    '''