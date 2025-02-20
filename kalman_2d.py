from filterpy.common import Q_discrete_white_noise, Saver
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
from acceleration_vector import find_acceleration_magnitude

def pos_vel_filter_2d(x, P, R, Q=0., dt=1.):
    """ Returns a KalmanFilter which implements a
    constant acceleration model for a state [x, y, vx, vy, ax, ay].T
    """

    kf = KalmanFilter(dim_x=6, dim_z=2)  # 6 state variables, 2 measurements

    # Initial state
    kf.x = np.array([x[0], x[1], x[2], x[3], x[4], x[5]])

    # State transition matrix (Constant Acceleration Model)
    kf.F = np.array([[1., 0., dt,  0., 0.5*dt**2, 0.],  # x = x + vx*dt + 0.5*ax*dt^2
                     [0., 1., 0.,  dt, 0., 0.5*dt**2],  # y = y + vy*dt + 0.5*ay*dt^2
                     [0., 0., 1.,  0., dt, 0.],         # vx = vx + ax*dt
                     [0., 0., 0.,  1., 0., dt],         # vy = vy + ay*dt
                     [0., 0., 0.,  0., 1., 0.],         # ax remains the same
                     [0., 0., 0.,  0., 0., 1.]])        # ay remains the same

    # Measurement function (We measure x and y directly)
    kf.H = np.array([[1., 0., 0., 0., 0., 0.], 
                     [0., 1., 0., 0., 0., 0.]])

    # Measurement noise covariance (sensor variance)
    kf.R = np.eye(2) * R if np.isscalar(R) else R

    # Initial covariance matrix
    kf.P = np.eye(6) * P if np.isscalar(P) else P

    # Process noise model (based on acceleration variance)
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q, block_size=2)

    return kf

def kalman_filter_2d(zs_x, zs_y, ta, times, smoothing=True):
    '''Takes x, y measurements and timestamps. Returns filtered data.'''

    times = pd.to_datetime(times)
    dt = times.diff().dt.total_seconds().mean()

    # Initial state [x, y, vx, vy, ax, ay]
    x = np.array([zs_x[0], zs_y[0], 0, 0, 0, 0])

    # Initial covariance matrices
    P = np.diag([0.2, 0.2, 9., 9., 5., 5.])  # Position & velocity uncertainty
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=20*dt, block_size=2)  # Process noise
    R = np.eye(2) * 0.2  # Measurement noise

    f = pos_vel_filter_2d(x, P, R, Q, dt)
    s = Saver(f)

    for i in range(len(zs_x)):
        f.predict()

        # Clamp estimated acceleration
        estimated_acc = np.linalg.norm(f.x[4:6])  # Magnitude of (ax, ay)
        if estimated_acc > ta[i]:
            f.x[4:6] *= ta[i] / estimated_acc  #THINK ABOUT THE BEST APPROACH HERE

        # Compute movement limit - might leave this to cullen's thing
        # if i >= 1:
        #     max_pos_change = np.linalg.norm(f.x[2:4]) * dt + 0.5 * ta[i] * dt**2
        #     predicted_change = np.linalg.norm([zs_x[i] - zs_x[i-1], zs_y[i] - zs_y[i-1]])

        #     if predicted_change > max_pos_change:
        #         zs_x[i] = zs_x[i-1] + np.sign(f.x[2]) * max_pos_change
        #         zs_y[i] = zs_y[i-1] + np.sign(f.x[3]) * max_pos_change


        #ADD INFLATED R BASED ON CONFIDENCE LEVELS HERE!!
        # Update filter with new measurement
        f.update(np.array([zs_x[i], zs_y[i]]))
        s.save()

    s.to_array()
    xs = s.x
    covs = s.P
    smooth_xs = None

    if smoothing:
        smooth_xs, _, _, _ = f.rts_smoother(xs, covs)

    return s, smooth_xs

def pipelineKalman_2d(df):
    df = df.copy()  
    df = find_acceleration_magnitude(df)  # Adds acceleration vectors to the df

    zs_x = df['pos_x'].values
    zs_y = df['pos_y'].values
    ta = df['ta'].values
    times = df['timestamp']

    s, smooth_xs = kalman_filter_2d(zs_x, zs_y, ta, times, smoothing=True)

    df['pos_x'] = s.x[:, 0]
    df['pos_y'] = s.x[:, 1]

    return df
