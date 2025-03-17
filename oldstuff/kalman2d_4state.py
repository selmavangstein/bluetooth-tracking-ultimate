from filterpy.kalman import KalmanFilter
import numpy as np
"""THIS FILE CAN BE DELETED"""

def pos_vel_filter_2d(x, P, R, Q=0., dt=1.):
    """
    Returns a KalmanFilter which implements a constant velocity model
    for a 2D state [x, y, vx, vy].
    
    x: initial state vector [x, y, vx, vy]
    P: initial covariance (scalar or 4x4)
    R: measurement noise (scalar or 2x2)
    Q: process noise variance (scalar). We'll build a Q with this 'var'
    dt: timestep
    """

    # 4 state variables, 2 measurements
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # Initial state [x, y, vx, vy]
    kf.x = np.array([x[0], x[1], x[2], x[3]])

    # State transition matrix for constant velocity:
    # x'  = x  + vx*dt
    # y'  = y  + vy*dt
    # vx' = vx
    # vy' = vy
    kf.F = np.array([[1., 0., dt, 0.],
                     [0., 1., 0., dt],
                     [0., 0., 1.,  0.],
                     [0., 0., 0.,  1.]])

    # Measurement function: we measure [x, y] directly
    kf.H = np.array([[1., 0., 0., 0.],
                     [0., 1., 0., 0.]])

    # Measurement noise covariance
    if np.isscalar(R):
        kf.R = np.eye(2) * R
    else:
        kf.R = R

    # Initial covariance
    if np.isscalar(P):
        kf.P = np.eye(4) * P
    else:
        kf.P = P

    # Process noise covariance Q:
    # We'll build it using the helper from FilterPy for a constant-velocity model.
    # Q_discrete_white_noise(dim=2, dt=dt, var=Q, block_size=2) yields a 4x4 block
    # for 2D (x,y) with constant velocity assumption.
    from filterpy.common import Q_discrete_white_noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q, block_size=2)

    return kf


import numpy as np
import pandas as pd
from filterpy.common import Q_discrete_white_noise, Saver

def kalman_filter_2d(zs_x, zs_y, times, confidence_factor, smoothing=True):
    """
    Takes x, y measurements, a 'ta' array (not used for acceleration anymore),
    timestamps, and confidence levels. Returns filtered data.

    The state is [x, y, vx, vy] with a constant-velocity model.
    """

    # Convert timestamps to pandas datetime and compute average dt
    times = pd.to_datetime(times)
    dt = times.diff().dt.total_seconds().mean()

    # Initial state: [x0, y0, vx=0, vy=0]
    x_init = np.array([zs_x[0], zs_y[0], 0, 0])

    # Covariance and noise
    # P: we assume some uncertainty in position and velocity
    P = np.diag([10, 10, 10.0, 10.0])  # tweak as needed
    # Q: process noise for constant-velocity model
    Q = Q_discrete_white_noise(dim=2, dt=dt, var=100*dt, block_size=2)
    # R: measurement noise
    R = np.eye(2) * 0.2

    # Build the Kalman filter with the CV model
    f = pos_vel_filter_2d(x_init, P, R, Q=Q, dt=dt)
    s = Saver(f)

    # For optional velocity clamp
    max_speed = 15.0

    impossible_circle_counter = 0

    for i in range(len(zs_x)):
        # Predict step
        f.predict()

        # No more acceleration clamp (since we don't model acceleration)
        # If you had logic for ta[i], you can remove or repurpose it.

        # Current measurement
        meas = np.array([zs_x[i], zs_y[i]])

        #2D velocity clamping. Can use ta here maybe too..?
        if i > 0:
            prev_coord = f.x[0:2]  # The filter's prior position estimate
            predicted_change = np.linalg.norm(meas - prev_coord)
            predicted_velocity = predicted_change / dt
            if predicted_velocity > max_speed:
                impossible_circle_counter += 1
                max_change = max_speed * dt
                direction = (meas - prev_coord) / predicted_change
                meas = prev_coord + max_change * direction

        # Dynamic scaling of measurement noise based on confidence
        dynamic_scale = 0.2 / (confidence_factor[i] + 1e-6)
        R_dynamic = R * dynamic_scale  # tweak the factor as needed
        f.R = R_dynamic

        # Update with the (possibly clamped) measurement
        f.update(meas)
        s.save()

        # Optionally clamp final position inside the field
        # Example: x in [-2,14], y in [-2,20]
        f.x[0] = np.clip(f.x[0], -2, 14)
        f.x[1] = np.clip(f.x[1], -2, 20)

    s.to_array()
    xs = s.x
    covs = s.P

    # Optional smoothing with RTS
    smooth_xs = None
    if smoothing:
        smooth_xs, _, _, _ = f.rts_smoother(xs, covs)

    print("Points corrected by impossible circle:", impossible_circle_counter)
    return s, smooth_xs


def pipelineKalman_2d(df):
    """
    Example pipeline function that:
    1) Possibly adds 'ta' and 'confidence' columns to the df.
    2) Runs the simplified Kalman filter with a CV model.
    3) Returns df with new pos_x, pos_y columns from the filter.
    """
    df = df.copy()

    # Suppose you have a function 'find_acceleration_magnitude' - 
    # it might be irrelevant now, but let's keep it if needed.
    # from acceleration_vector import find_acceleration_magnitude
    # df = find_acceleration_magnitude(df)

    zs_x = df['pos_x'].values
    zs_y = df['pos_y'].values
    times = df['timestamp']
    confidence_factor = df['confidence']

    s, smooth_xs = kalman_filter_2d(zs_x, zs_y, times, confidence_factor, smoothing=True)

    # Overwrite df columns with filtered data
    df['pos_x'] = s.x[:, 0]
    df['pos_y'] = s.x[:, 1]

    return df
