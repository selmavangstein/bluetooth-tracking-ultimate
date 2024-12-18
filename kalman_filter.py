from filterpy.kalman import KalmanFilter
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

f = KalmanFilter(dim_x=2, dim_z=1) #we track distance from beacon, and also velocity (to predict next position)

#initial state
f.x = np.array([2., 0.]) #[position, velocity]

#we now set a bunch of matrices. Need to figure out how to calibrate the matrices

#state transition matrix
f.F = np.array([[1.,1.],
                [0.,1.]])

#measurement fcn
f.H = np.array([[1.,0.]])

#covariance matrix
f.P = np.array([[1000.,    0.],
                [   0., 1000.] ]) #uncertainty is 1000..?

#measurement noise
f.R = 5

#process noise
from filterpy.common import Q_discrete_white_noise
f.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

df_data = pd.read_csv("makecharts\walking test to 60_cleaned.csv")
kalmanDistace = []

#replace condition - do like a read from our measurement
for i in range(len(df_data["distance"])):
    z = df_data["distance"][i] #replace with reading from the data file. z=distance
    f.predict()
    f.update(z)

    kalmanDistace.append(f.x)

df_data['timestamp'] = pd.to_datetime(df_data['timestamp'])

plt.figure(figsize=(12, 8))
plt.plot(df_data["timestamp"], df_data["distance"], label="og measurement")
plt.plot(df_data["timestamp"], kalmanDistace, label="kalman")
plt.xlabel("timestamp")
plt.ylabel("distance (m)")
plt.title("kalman filtered data")
plt.legend()
plt.show()