import numpy as np
import itertools
import matplotlib.pyplot as plt
import pandas as pd

def trilaterate_one(beacons, distances):
    print("beacons received by trilaterate one: ", beacons)
    P1, P2, P3 = beacons
    r1, r2, r3 = distances
    
    P21 = P2 - P1
    P31 = P3 - P1
    
    A = 2 * np.array([
        [P21[0], P21[1]],
        [P31[0], P31[1]]
    ])
    
    b = np.array([
        r1**2 - r2**2 - np.dot(P1, P1) + np.dot(P2, P2),
        r1**2 - r3**2 - np.dot(P1, P1) + np.dot(P3, P3)
    ])
    
    try:
        position = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    return position

def circle_intersections(p1, r1, p2, r2):
    """Finner skjæringspunktene mellom to sirkler gitt deres sentre og radier."""
    d = np.linalg.norm(p2 - p1)
    
    # Ingen skjæring
    if d > r1 + r2 or d < abs(r1 - r2):
        return []

    # Beregn punkt på linjen mellom sentrene hvor skjæringspunktene ligger
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h = np.sqrt(max(0, r1**2 - a**2))

    # Finn midtpunkt på linjen mellom sirklene
    p_mid = p1 + a * (p2 - p1) / d

    # Finn skjæringspunktene
    offset = h * np.array([-(p2[1] - p1[1]) / d, (p2[0] - p1[0]) / d])
    return [p_mid + offset, p_mid - offset]

def find_best_trilateration(beacons, distances):
    """Finner de tre skjæringspunktene som er nærmest hverandre og bruker til trilaterasjon."""
    intersections = []
    intersection_to_beacons = {}

    # Finn alle skjæringspunkter mellom sirklene
    for (i, j) in itertools.combinations(range(len(beacons)), 2):
        p1, r1 = beacons[i], distances[i]
        p2, r2 = beacons[j], distances[j]
        points = circle_intersections(p1, r1, p2, r2)
        
        for point in points:
            intersections.append(point)
            intersection_to_beacons[tuple(point)] = (i, j)

    if len(intersections) < 3:
        raise ValueError("For få skjæringspunkter funnet")

    # Beregn avstandsmatrise mellom alle skjæringspunktene
    intersections = np.array(intersections)
    pairwise_distances = np.linalg.norm(intersections[:, np.newaxis] - intersections, axis=-1)

    # Finn de tre punktene som er nærmest hverandre
    best_indices = np.argsort(np.sum(pairwise_distances, axis=1))[:3]
    best_points = intersections[best_indices]

    # Finn de tre beaconene som skapte disse skjæringspunktene
    beacon_indices = set()
    for point in best_points:
        beacon_indices.update(intersection_to_beacons[tuple(point)])

    if len(beacon_indices) < 3:
        raise ValueError("For få unike beacons funnet")

    best_beacon_indices = np.array(list(beacon_indices))[:3]  # Ensure exactly 3 beacons
    best_beacons = beacons[best_beacon_indices]
    best_distances = distances[best_beacon_indices]

    # Bruk trilaterasjon på de tre beste beaconene
    return trilaterate_one(best_beacons, best_distances), best_points

def plot_intersections(beacons, distances, intersections, final_position):
    """Plotter beacons, sirkler, skjæringspunkter og den estimerte posisjonen."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Tegn sirkler for beacons
    for i, (beacon, radius) in enumerate(zip(beacons, distances)):
        circle = plt.Circle(beacon, radius, color=f"C{i}", alpha=0.3, fill=False)
        ax.add_patch(circle)
        ax.scatter(*beacon, color=f"C{i}", label=f"Beacon {i+1}")
        ax.text(beacon[0], beacon[1], f" B{i+1}", fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # Plot skjæringspunkter
    intersections = np.array(intersections)
    ax.scatter(intersections[:, 0], intersections[:, 1], color="black", marker="x", label="Intersection points")

    # Plot endelig posisjon
    ax.scatter(*final_position, color="red", marker="*", s=200, label="Final estimated position")

    ax.set_xlim(beacons[:, 0].min() - 2, beacons[:, 0].max() + 2)
    ax.set_ylim(beacons[:, 1].min() - 2, beacons[:, 1].max() + 2)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Best Trilateration Based on Circle Intersections")
    ax.legend()
    ax.set_aspect("equal")
    
    plt.show()


testdata = pd.read_csv("processedtest6.csv")
row = testdata.iloc[100]
distances = np.array(row[[col for col in testdata.columns if col.startswith('b')]])
beacons = np.array([[0, 0], [12, 0], [0, 16], [12, 16]])
# Finn beste trilaterasjon basert på skjæringspunktene
final_position, intersections = find_best_trilateration(beacons, distances)

# Plott resultatet
plot_intersections(beacons, distances, intersections, final_position)



