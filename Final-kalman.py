
from filterpy.stats import plot_covariance_ellipse
from filterpy.common import Q_discrete_white_noise
import numpy as np
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.kalman import KalmanFilter

def kalmanFilter(df, x=np.array([10.0, 0]), P=np.diag([30, 16]), R=np.array([[5.]]), Q=Q_discrete_white_noise(dim=2, dt=0.3, var=2.35), dt=0.3):
    """
    Applies the Kalman filter to every column in the dataframe that starts with 'b'.
    """
    def pos_vel_filter(x, P, R, Q=0., dt=1.):
        """ Returns a KalmanFilter which implements a
        constant velocity model for a state [x dx].T
        """

        #the ones that are explicitly set in this fcn are probably less likely to be adjusted
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
    
    def run(x0=(0.,0.), P=500, R=0, Q=0, dt=1.0, 
        track=None, zs=None,
        count=0, do_plot=True, **kwargs):
        """
        track is the actual position of the dog, zs are the 
        corresponding measurements. 
        """

        # Simulate dog if no data provided. 
        if zs is None:
            print("no data provided, cannot run filter")
            return False

        # create the Kalman filter
        kf = pos_vel_filter(x0, R=R, P=P, Q=Q, dt=dt)  

        # run the kalman filter and store the results
        xs, cov = [], []
        for z in zs:
            kf.predict() #predicts next position
            kf.update(z) #takes next measurement, updates position
            xs.append(kf.x) #stores result
            cov.append(kf.P) #stores result

        xs, cov = np.array(xs), np.array(cov)
        return xs, cov

    df = df.copy()
    results = {}
    for column in df.columns:
        if column.startswith('b'):
            zs = df[column].values
            xs, cov = run(x, P, R, Q, dt=dt, zs=zs)
            results[column] = xs[:, 0]  # store the position estimates

    # Add the results to the dataframe, replacing the original data
    for column, values in results.items():
        df[column] = values

    return df


def main():
    # test here
    pass

if __name__ == "__main__":
    main()