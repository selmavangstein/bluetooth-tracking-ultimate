from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import numpy as np
from filterpy.kalman import KalmanFilter

""" Kalman Filter on measured position and acceleration vectors with no added restriction steps."""

def pos_vel_acc_filter(x, P, R, Q=0., dt=1.):
    """ Initializes a Kalman filter on position and acceleration measurements with velocity as a hidden variable.
    Args:
        x: Initial state
        P: Initial state uncertainty
        R: Measurement uncertainty
        Q: Process noise variance
        dt: Time difference between each data point
    Returns:
        kf: A Kalman filter with the given controls
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
    '''Applies a Kalman filter to the given data. 
    Args:
        pos_data: Position measurements to filter
        acc_data: Acceleration vectors
        times: list of timestamps
        smoothing: Boolean to apply RTS smoothing.
    Returns:
        s: saver object storing the state history of the filter, including filtered distances
 '''
    time_differences = times.diff()
    average_difference = time_differences.dt.total_seconds().mean()

    dt = average_difference

    x = np.array([10.0, 0, 0]) #initial state
    P = np.diag([70, 16, 4]) #initial state uncertainty 
    R = np.array([[3., 0.],
                [0., 5.]]) #measurement covariance matrix /sensor variance
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=2.35) #process noise 

    zs = list(zip(pos_data, acc_data))

    #initialize filter
    f = pos_vel_acc_filter(x, P, R, Q, dt)
    s = Saver(f)

    #filter cycle
    xs, covs, _, _ = f.batch_filter(zs, saver=s)
    smooth_xs = None
    if smoothing:
        smooth_xs, smooth_cov, _, _ = f.rts_smoother(xs, covs)
    s.to_array()
    return s, smooth_xs