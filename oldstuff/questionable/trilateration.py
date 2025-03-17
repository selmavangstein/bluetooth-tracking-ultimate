import numpy as np


def trilaterate_one(beacons, distances):
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

def trilateration_two(beacon1, S1, beacon2, S2, beacon3, S3):
    #Extracting the x and y coordinates of beacon. 
    x1, y1 = beacon1
    x2, y2 = beacon2
    x3, y3 = beacon3

    # Calculating constants used to simplify the equations derived from the distance formula. 
    # These constants are intermediate values used to form linear equations
    a = 2 * (x2 - x1)
    b = 2 * (y2 - y1)
    c = 2 * (x3 - x2)
    d = 2 * (y3 - y2)    

    # The constants e and f are derived by reducing the distance equations for each beacon. 
    e = S1**2 - S2**2 - x1**2 + x2**2 - y1**2 + y2**2
    f = S2**2 - S3**2 - x2**2 + x3**2 - y2**2 + y3**2

    # Calculating the denominator for the equations. 
    denominator = (a**2 + c**2) * (b**2 + d**2) - (a * b + c * d)**2

    # Checking if the denominator is zero, which would indicate invalid beacon positions. 
    if denominator == 0:
        raise ValueError("The beacons do not form a valid triangle for trilateration.")

    # Solving for the x and y coordinate of the unknown point using the derived equations. 
    x = ((b**2 + d**2) * (a * e + c * f) - (a * b + c * d) * (b * e + d * f)) / denominator
    y = ((a**2 + c**2) * (b * e + d * f) - (a * b + c * d) * (a * e + c * f)) / denominator

    return (x, y)

# Example usage (one)
def example_usage():
    # Define three beacon positions
    beacons = np.array([
        [0, 0],    # Beacon 1 at origin
        [30, 0],   # Beacon 2 at (10,0)
        [15, 26]    # Beacon 3 at (0,10)
    ])
    
    # Define distances from each beacon to the target
    distances = np.array([17, 18, 15])
    
    try:
        position = trilaterate(beacons, distances)
        print(f"Calculated position: ({position[0]:.2f}, {position[1]:.2f})")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    example_usage()
    