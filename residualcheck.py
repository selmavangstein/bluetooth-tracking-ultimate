import numpy as np

"""CAN BE DELETED - NOT CALLED ANYWHERE"""

def residualcheck(s, R):
    """Finds the residual variance in a set of residuals and prints it.
    Useful in helping tuning the Kalman filter"""
    residuals = np.array(s.y)
    residual_variance = np.var(residuals)

    print(f"Residual Variance: {residual_variance}")