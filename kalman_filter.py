from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
import numpy as np
from filterpy.kalman import KalmanFilter
import pandas as pd
#from kf_book.mkf_internal import plot_track
from kalman_plotting import plot_results
import matplotlib.pyplot as plt


def pos_vel_filter(x, P, R, Q=0., dt=1.):
    """ Returns a KalmanFilter which implements a
    constant velocity model for a state [x dx].T
    """

    kf = KalmanFilter(dim_x=2, dim_z=1)
    s = Saver(kf)
    kf.x = np.array([x[0], x[1]]) # location and velocity. 
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
    return kf, s

def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0, zs=None, make_plot=False, actual=None):

    if zs is None:
        print("no data provided, cannot run filter")
        return False

    # create the Kalman filter
    kf, s = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)  

    # run the kalman filter and store the results
    for z in zs:
        #can I update dt here?
        kf.predict() #predicts next position
        kf.update(z) #takes next measurement, updates position
        s.save()

    if make_plot:
        print("creating plots")
        s.to_array()
        plot_results(s.x[:, 0], s.z, s.P)
    return s

dt = 0.3 #compute by subtracting timestamps
#Use initial measurement to set x0. Can take x1-x0=v0
x = np.array([10.0, 0]) #initial state.
P = np.diag([30, 16]) #initial state uncertainty
Q = Q_discrete_white_noise(dim=2, dt=dt, var=2.35) #process noise
R = np.array([[5.]]) #measurement covariance matrix /sensor variance.

df_data = pd.read_csv("bluetooth-tracking-ultimate\makecharts\walking test to 60_cleaned.csv")
zs = df_data["distance"]

s = run(x, P, R, Q, dt=dt, zs=zs, make_plot=True)

""" ps = xs[:, 0]
count = len(ps)
plt.figure()
plt.plot(range(count), zs)
plt.plot(range(count), ps)
plt.title("pos vs time")
plt.show() """
