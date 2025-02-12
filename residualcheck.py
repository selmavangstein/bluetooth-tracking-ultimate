import numpy as np

def residualcheck(s, R):
    residuals = np.array(s.y)  # Extract measurement residuals
    residual_variance = np.var(residuals)

    print(f"Residual Variance: {residual_variance}")
    print(f"Current R: {R[0,0]}")  # Assuming R is scalar

    # Compare and adjust R if needed
    if residual_variance > R[0, 0]:
        print("R might be too low; consider increasing it.")
    elif residual_variance < R[0, 0]:
        print("R might be too high; consider decreasing it.")
    else:
        print("R looks well-tuned!")