from filterpy.common import Q_discrete_white_noise, Saver
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
from acceleration_vector import find_acceleration_magnitude
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""This file implements a Kalman filter to filter a 2D state, in our use case a position coordinate.
Hidden variables are velocity and acceleration.
We use this repo, which is very well documented: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python"""

def pos_vel_filter_2d(x, P, R, Q=0., dt=1.):
    """ Returns a KalmanFilter which implements a constant acceleration model for a 2D state.
    Args:
        x: Initial state
        P: Initial state uncertainty
        R: Measurement uncertainty
        Q: Process noise variance
        dt: Time difference between each data point
    Returns:
        kf: A Kalman filter with the given controls   
    """

    kf = KalmanFilter(dim_x=6, dim_z=2)  #6 state variables, 2 measurements

    #Initial state
    kf.x = np.array([x[0], x[1], x[2], x[3], x[4], x[5]])

    #State transition matrix (Constant Acceleration Model)
    kf.F = np.array([[1., 0., dt,  0., 0.5*dt**2, 0.],  #x = x + vx*dt + 0.5*ax*dt^2
                     [0., 1., 0.,  dt, 0., 0.5*dt**2],  #y = y + vy*dt + 0.5*ay*dt^2
                     [0., 0., 1.,  0., dt, 0.],         #vx = vx + ax*dt
                     [0., 0., 0.,  1., 0., dt],         #vy = vy + ay*dt
                     [0., 0., 0.,  0., 1., 0.],         #ax remains the same
                     [0., 0., 0.,  0., 0., 1.]])        #ay remains the same

    #Measurement function (We measure x and y directly)
    kf.H = np.array([[1., 0., 0., 0., 0., 0.], 
                     [0., 1., 0., 0., 0., 0.]])

    #Measurement noise covariance (sensor variance)
    kf.R = np.eye(2) * R if np.isscalar(R) else R

    #Initial covariance matrix
    kf.P = np.eye(6) * P if np.isscalar(P) else P

    #Process noise model
    kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q, block_size=2)

    return kf

def kalman_filter_2d(zs_x, zs_y, ta, times, confidence_factor, beacon_pos, smoothing=True, ani=False):
    '''Applies a Kalman filter to the given data.
    Args:
        zs_x: x-coordinate to filter
        zs_y: y-coordinate to filter
        ta: Acceleration magnitude
        times: list of timestamps
        confidence_factor: Confidence/uncertainty in the given coordinate
        beacon_pos: List of beacon coordinates
        smoothing: Boolean to apply RTS smoothing.
        ani: Boolean to indicate whether to create an animation of the filter
    Returns:
        s: saver object storing the state history of the filter, including filtered distances
        smooth_xs: Smoothed coordinates
 '''

    times = pd.to_datetime(times, format='%H:%M:%S.%f')
    dt = times.diff().dt.total_seconds().mean()

    #field bounds
    x_min, y_min = np.min(beacon_pos, axis=0)
    x_max, y_max = np.max(beacon_pos, axis=0)

    x = np.array([zs_x[0], zs_y[0], 0, 0, 0, 0]) #Initial state [x, y, vx, vy, ax, ay]
    P = np.diag([0.2, 0.2, 9., 9., 5., 5.])  # Initial state uncertainty
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=200*dt, block_size=2)  # Process noise
    R = np.eye(2) * 0.8  # Measurement noise

    #initialize the filter
    f = pos_vel_filter_2d(x, P, R, Q, dt)
    s = Saver(f)

    #initialize additional restrictions
    impossible_circle_counter = 0
    inflated_R_counter = 0
    measurements = []
    pred_states = []
    filt_states = []
    #main filter cycle
    for i in range(len(zs_x)):
        prev_coord = f.x[0:2]
        f.predict()
        pred_states.append(f.x.copy())

        # Clamp estimated acceleration
        estimated_acc = np.linalg.norm(f.x[4:6])  #Magnitude of (ax, ay)
        if estimated_acc > ta[i]:
            f.x[4:6] *= ta[i] / estimated_acc #Scale back the acceleration if it is too high to be realistic

        #clamp dx/dt (velocity)
        meas = np.array([zs_x[i], zs_y[i]])
        measurements.append(meas.copy())
        max_vel = 11
        if i > 0:
            predicted_change = np.linalg.norm(meas - prev_coord)
            predicted_velocity = predicted_change/dt
            if predicted_velocity > max_vel:
                impossible_circle_counter+=1
                max_change = max_vel * dt  # maximum allowed displacement
                direction = (meas - prev_coord) / predicted_change
                meas = prev_coord + max_change * direction

        #clamp velocity prediction
        speed = np.linalg.norm(f.x[2:4])
        if speed > max_vel:
            f.x[2:4] *= max_vel / speed

        #inflate measurement uncertainty if the coordinate confidence is too low
        cutoff = 0.5          #Confidence threshold
        inflation = 100.0
        if confidence_factor[i] < cutoff:
            R_dynamic = R * inflation
            inflated_R_counter += 1
        else:
            R_dynamic = R

        f.R = R_dynamic

        # Update filter with new measurement
        f.update(meas)
        filt_states.append(f.x.copy())

        #Limit coordinate to be within 2 meters of the field
        f.x[0] = np.clip(f.x[0], x_min-2, x_max+2)
        f.x[1] = np.clip(f.x[1], y_min-2, y_max+2)
        s.save()

    s.to_array()
    xs = s.x
    covs = s.P
    smooth_xs = None
    if smoothing:
        smooth_xs, _, _, _ = f.rts_smoother(xs, covs)

    print("Points corrected by impossible circle: ", impossible_circle_counter)
    print("inflated R's: ", inflated_R_counter)

    if ani:
        animateKalman(pred_states, measurements, filt_states, smooth_xs, x_min, x_max, y_min, y_max)

    return s, smooth_xs

def pipelineKalman_2d(df, beacon_pos):
    """Applies a Kalman filter to position coordinates from dataframe
    Args:
        df: dataframe with position coordinates, acceleration data, timestamps and confidence values
        beacon_pos: Array of beacon coordinates
    Returns:
        df: copy of df with filtered position coordinates
    """
    df = df.copy()  
    df = find_acceleration_magnitude(df)  #Adds acceleration magnitudes to the df

    #extract data from dataframe
    zs_x = df['pos_x'].values
    zs_y = df['pos_y'].values
    ta = df['ta'].values
    times = df['timestamp']
    confidence_factor = df['confidence']

    #filter data
    s, smooth_xs,  = kalman_filter_2d(zs_x, zs_y, ta, times, confidence_factor, beacon_pos, smoothing=True, ani=False)

    #add filtered data to dataframe
    # df['pos_x'] = s.x[:, 0]
    # df['pos_y'] = s.x[:, 1]
    df['pos_x'] = smooth_xs[:, 0]
    df['pos_y'] = smooth_xs[:, 1]

    return df

def animateKalman(pred_states, measurements, filt_states, smooth_states):
    """Creates an animation that shows how the filter works over the duration of the test.
    Args:
        pred_states: filter predictions
        measurements: coordinates before filtering
        filt_states: coordinates after filtering
        smooth_states: final smoothed coordinates (after filtering)
    """
    #Convert lists to numpy arrays for easier slicing
    measurements = np.array(measurements)
    pred_states = np.array(pred_states)
    filt_states = np.array(filt_states)
    smooth_states = np.array(smooth_states)

    #Set up the figure
    gt = pd.read_csv('data/GT-obstacletest2.csv')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(3, 16)
    plt.xticks([3,6,9])
    plt.yticks([6,12])
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.plot(gt['locx'], gt['locy'], '-', color='grey', label='Ground Truth')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Kalman 2D Filter Animation")

    #Create markers for each tracked state
    measurement_point, = ax.plot([], [], 'rx', markersize=10, label="Measurement")
    prediction_point, = ax.plot([], [], 'bo', markersize=8, label="Prediction")
    filtered_point, = ax.plot([], [], '^', color='orange', markersize=8, label="Filtered")
    smooth_point, = ax.plot([], [], 'gP', markersize=8, label="Smoothed")
    trajectory_line, = ax.plot([], [], 'g-', linewidth=1, label="Trajectory")

    ax.legend()

    def init():
        """initializes objects to store animation frames"""
        measurement_point.set_data([], [])
        prediction_point.set_data([], [])
        filtered_point.set_data([], [])
        smooth_point.set_data([], [])
        trajectory_line.set_data([], [])
        return measurement_point, prediction_point, filtered_point, smooth_point, trajectory_line

    def update(frame):
        """Updates each frame of the animation"""
        meas = measurements[frame]
        pred = pred_states[frame]
        filt = filt_states[frame]
        smooth = smooth_states[frame]
        
        #Wrap coordinates in a list so that set_data gets a sequence
        measurement_point.set_data([meas[0]], [meas[1]])
        prediction_point.set_data([pred[0]], [pred[1]])
        filtered_point.set_data([filt[0]], [filt[1]])
        smooth_point.set_data([smooth[0]], [smooth[1]])
        
        #For trajectory, plot all filtered positions up to current frame
        traj = smooth_states[:frame+1]
        trajectory_line.set_data(traj[:, 0], traj[:, 1])
        
        ax.set_title(f"Kalman 2D Filter - Step {frame}")
        return measurement_point, prediction_point, filtered_point, smooth_point, trajectory_line


    #Create the animation
    anim = animation.FuncAnimation(fig, update, frames=len(measurements),
                                init_func=init, blit=True, interval=80)
    #Save animation
    anim.save('kalman_animation.gif', writer='pillow', fps=30)