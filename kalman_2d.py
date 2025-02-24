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

def kalman_filter_2d(zs_x, zs_y, ta, times, confidence_factor, smoothing=True):
    '''Takes x, y measurements and timestamps. Returns filtered data.'''

    times = pd.to_datetime(times)
    dt = times.diff().dt.total_seconds().mean()

    # Initial state [x, y, vx, vy, ax, ay]
    x = np.array([zs_x[0], zs_y[0], 0, 0, 0, 0])

    # Initial covariance matrices
    P = np.diag([0.2, 0.2, 9., 9., 5., 5.])  # Position & velocity uncertainty
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=200*dt, block_size=2)  # Process noise
    R = np.eye(2) * 0.8  # Measurement noise

    max_vel = 11

    f = pos_vel_filter_2d(x, P, R, Q, dt)
    s = Saver(f)

    impossible_circle_counter = 0
    inflated_R_counter = 0
    for i in range(len(zs_x)):
        prev_coord = f.x[0:2]
        f.predict()

        # Clamp estimated acceleration
        estimated_acc = np.linalg.norm(f.x[4:6])  # Magnitude of (ax, ay)
        if estimated_acc > ta[i]:
            f.x[4:6] *= ta[i] / estimated_acc

        meas = np.array([zs_x[i], zs_y[i]])

        if i > 0:
            predicted_change = np.linalg.norm(meas - prev_coord)
            predicted_velocity = predicted_change/dt
            if predicted_velocity > max_vel:
                impossible_circle_counter+=1
                max_change = max_vel * dt  # maximum allowed displacement
                direction = (meas - prev_coord) / predicted_change
                meas = prev_coord + max_change * direction

        speed = np.linalg.norm(f.x[2:4])
        if speed > max_vel:
            f.x[2:4] *= max_vel / speed


        # Option 1: Dynamic scaling based on confidence (continuous approach)
        # We assume confidence_factor[i] is between 0 and 1.
        # Lower confidence → higher measurement noise.
        #dynamic_scale = 0.2 / (confidence_factor[i] + 1e-6) #TRY EXPONENTIAL
        #R_dynamic = R * dynamic_scale

        # Option 2: Hard cutoff (uncomment this block if you prefer cutoff behavior)
        cutoff = 0.5          # Confidence threshold
        inflation = 100.0     # How much to inflate R if below threshold
        if confidence_factor[i] < cutoff:
            R_dynamic = R * inflation
            inflated_R_counter += 1
        else:
            R_dynamic = R

        f.R = R_dynamic

        # Update filter with new measurement
        f.update(meas)
        s.save()

        #fix this hardcoding
        f.x[0] = np.clip(f.x[0], -2, 14)
        f.x[1] = np.clip(f.x[1], -2, 20)

    s.to_array()
    xs = s.x
    covs = s.P
    smooth_xs = None

    if smoothing:
        smooth_xs, _, _, _ = f.rts_smoother(xs, covs)

    print("Points corrected by impossible circle: ", impossible_circle_counter)
    print("inflated R's: ", inflated_R_counter)
    return s, smooth_xs

def pipelineKalman_2d(df):
    df = df.copy()  
    df = find_acceleration_magnitude(df)  # Adds acceleration vectors to the df

    zs_x = df['pos_x'].values
    zs_y = df['pos_y'].values
    ta = df['ta'].values
    times = df['timestamp']
    confidence_factor = df['confidence']

    s, smooth_xs = kalman_filter_2d(zs_x, zs_y, ta, times, confidence_factor, smoothing=True)

    df['pos_x'] = s.x[:, 0]
    df['pos_y'] = s.x[:, 1]

    return df