from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
from collections import deque
from acceleration_vector import find_acceleration_magnitude

"""This file implements a Kalman filter. The position state is the only needed measurement, 
velocity and acceleration are hidden variables. 
pipelineKalman is the function that is used to apply the filter to the distance measurements from tests.
We ended up not using this filter as it did nothing we could not do with other algorithms, but we believe
this can be helpful in the future with better tuning, more complex modelling, or another measured variable."""

def pos_vel_acc_filter(x, P, R, Q=0., dt=1.):
    """ Initializes a Kalman filter with position as measured variable and velocity and accceleration as
    hidden variables.
    Uses a standard kinematic model with constant acceleration.
    Args:
        x: Initial state
        P: Initial state uncertainty
        R: Measurement uncertainty
        Q: Process noise variance
        dt: Time difference between each data point
    Returns:
        kf: A Kalman filter with the given controls
    """

    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.array([x[0], x[1], x[2]]) #initial state
    kf.F = np.array([[1., dt, 1/2*dt**2],
                     [0.,  1., dt],
                     [0., 0., 1]])  #transition matrix
    kf.H = np.array([[1., 0, 0.]])    #Measurement function
    if np.isscalar(R):
        kf.R *= R                 #covariance matrix 
    else:
        kf.R[:] = R                     #measurement uncertainty
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
    '''Applies a Kalman filter to the given data.
    Args:
        zs: Measurement to filter
        ta: Acceleration magnitude
        times: list of timestamps
        smoothing: Boolean to apply RTS smoothing.
    Returns:
        s: saver object storing the state history of the filter, including filtered distances
 '''

    times = pd.to_datetime(times, format='%H:%M:%S.%f')
    time_differences = times.diff()
    average_difference = time_differences.dt.total_seconds().mean()
    dt = average_difference

    #set up filter matrices. It is important to tune these to the specific use case
    x = np.array([zs[0], 0, 0]) #initial state.
    P = np.diag([.2, 9., 5.]) #initial state uncertainty 
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=20*dt) #process noise.
    R = np.array([[0.2]]) #measurement covariance matrix /sensor variance.

    #initialize filter
    f = pos_vel_acc_filter(x, P, R, Q, dt) 
    s = Saver(f)

    #gating parameters
    gating_threshold = 3 #how many std devs to consider an outlier
    R_inflated = 50.0
    inflation_steps = 5
    inflation_steps_countdown = inflation_steps
    R_original = R.copy()

    #rolling window params
    window_size = 10
    residual_window = deque([], maxlen=window_size)
    variance_threshold = 2.0  # tune this based on typical residual variance - FIND THIS

    #some counts for testing purposes
    delta_pos_too_high = 0
    acc_too_high = 0
    stdev_too_high = 0
    residual_var_too_high = 0

    #main filter cycle
    for i in range(len(zs)):
        f.predict()
        z = zs[i]

        #clamp esitmated acceleration
        estimated_acc = abs(f.x[2])
        if estimated_acc > ta[i]:
            acc_too_high += 1  
            f.x[2] = np.sign(f.x[2]) * ta[i]

        #handle nan-values
        if np.isnan(z):
            z=f.x[0]
            s.save()
            s.x[-1][0] = np.nan
            s.x[-1][1] = np.nan
            continue                

        #compute innovation and innovation covariance S
        innovation = z - f.x[0] #innovation is difference between prediction and measurement
        S = f.P[0, 0] + f.R[0, 0] #initial covariance plus current covariance
        stdev = np.sqrt(S)

        if abs(innovation) > gating_threshold * stdev:
            stdev_too_high += 1

            #Temporarily inflate R
            f.R = R_original * R_inflated
            inflation_steps_countdown = inflation_steps
        else:
            #If not an outlier, check if we are still in "inflated" mode
            if inflation_steps_countdown > 0:
                inflation_steps_countdown -= 1
                if inflation_steps_countdown == 0:
                    # revert to normal R
                    f.R = R_original
            else:
                #normal operation
                f.R = R_original

        f.update(z)
        s.save()

        #if the variance in the residuals in a window is too high we inflate R
        new_residual = s.y[-1]
        residual_window.append(new_residual)
        if len(residual_window) == window_size: #residual window has a max size, works as a queue
            var_estimate = np.var(residual_window)
            if var_estimate > variance_threshold:
                residual_var_too_high += 1
                f.R = R_original * R_inflated
                inflation_steps_countdown = inflation_steps

    print(f"Acceleration clamped: {acc_too_high}")
    print(f"Position change clamped: {delta_pos_too_high}")
    print(f"R inflated due to stdev: {stdev_too_high}")
    print(f"R inflated due to residual var: {residual_var_too_high}")
    
    s.to_array()
    xs = s.x
    covs = s.P
    smooth_xs = None
    if smoothing:
        smooth_xs, smooth_cov, _, _ = f.rts_smoother(xs, covs)

    #this returns a saver object with all the information about the filter
    return s, smooth_xs


def pipelineKalman(df):
    """Applies a Kalman filter to a one dimensional measurement (distances in our usecase)
    Args:
        df: dataframe containing timestamps, distance measurements, acceleration data
    Returns:
        df: a copy of df with filtered distances, and a new column 'ta' containing acceleration magnitudes
    """

    df = df.copy()  
    df = find_acceleration_magnitude(df) # adds acceleration vectors to the df

    #filter each distance column
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