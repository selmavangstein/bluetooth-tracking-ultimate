import numpy as np

def residualcheck(s, R):
    residuals = np.array(s.y)  # Extract measurement residuals
    residual_variance = np.var(residuals)

    print(f"Residual Variance: {residual_variance}")