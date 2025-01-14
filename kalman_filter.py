from filterpy.common import Q_discrete_white_noise
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
#from kf_book.mkf_internal import plot_track
import matplotlib.pyplot as plt


def pos_vel_filter(x, P, R, Q=0., dt=1.):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([x[0], x[1]]) # location and velocity
    kf.F = np.array([[1., dt],
                     [0.,  1.]])  # state transition matrix
    kf.H = np.array([[1., 0]])    # Measurement function
    kf.R *= R                     # measurement uncertainty
    if np.isscalar(P):
        kf.P *= P                 # covariance matrix 
    else:
        kf.P[:] = P               # [:] makes deep copy
    if np.isscalar(Q):
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q)
    else:
        kf.Q[:] = Q
    return kf

def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0, zs=None):

    if zs is None:
        print("no data provided, cannot run filter")
        return False

    # create the Kalman filter
    kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)  

    # run the kalman filter and store the results
    xs, cov = [], []
    for z in zs:
        #can I update dt here?
        kf.predict() #predicts next position
        kf.update(z) #takes next measurement, updates position
        xs.append(kf.x) #stores result
        cov.append(kf.P) #stores result

    xs, cov = np.array(xs), np.array(cov)
    return xs, cov

dt = 0.3 
x = np.array([10.0, 0]) #initial state
P = np.diag([30, 16]) #initial state uncertainty
Q = Q_discrete_white_noise(dim=2, dt=dt, var=2.35) #process noise
R = np.array([[5.]]) #measurement covariance matrix /sensor variance.

df_data = pd.read_csv("bluetooth-tracking-ultimate\makecharts\walking test to 60_cleaned.csv")
zs = df_data["distance"]

xs, cov = run(x, P, R, Q, dt=dt, zs=zs)

ps = xs[:, 0]
count = len(ps)
plt.plot(range(count), zs)
plt.plot(range(count), ps)
plt.show()