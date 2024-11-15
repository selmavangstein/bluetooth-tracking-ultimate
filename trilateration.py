import numpy as np

def trilaterate(beacons, distances):
    """
    Determine the position of a point using trilateration from three known points and their distances.
    
    Parameters:
    beacons: numpy array of shape (3, 2) containing the x,y coordinates of three beacons
    distances: numpy array of shape (3,) containing the distances from each beacon to the target point
    
    Returns:
    numpy array of shape (2,) containing the x,y coordinates of the calculated position
    """
    # Extract individual beacon coordinates
    P1, P2, P3 = beacons
    r1, r2, r3 = distances
    
    # Calculate vectors between points
    P21 = P2 - P1
    P31 = P3 - P1
    
    # Create coefficients matrix A and vector b for the equation Ax = b
    A = 2 * np.array([
        [P21[0], P21[1]],
        [P31[0], P31[1]]
    ])
    
    b = np.array([
        r1*r1 - r2*r2 - np.dot(P1, P1) + np.dot(P2, P2),
        r1*r1 - r3*r3 - np.dot(P1, P1) + np.dot(P3, P3)
    ])
    
    # Solve the system of equations
    try:
        position = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("The beacons' positions don't allow for a unique solution")
    
    return position

# Example usage
def example_usage():
    # Define three beacon positions
    beacons = np.array([
        [0, 0],    # Beacon 1 at origin
        [10, 0],   # Beacon 2 at (10,0)
        [0, 10]    # Beacon 3 at (0,10)
    ])
    
    # Define distances from each beacon to the target
    distances = np.array([5, 7.07, 7.07])
    
    try:
        position = trilaterate(beacons, distances)
        print(f"Calculated position: ({position[0]:.2f}, {position[1]:.2f})")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    example_usage()
    