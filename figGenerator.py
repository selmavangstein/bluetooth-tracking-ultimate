import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('player_positions_Velocity Clamping4.csv')
gt = pd.read_csv('data/GT-obstacletest2.csv')
beacons = np.array([[0, 0], [12, 0], [0, 18], [12, 18]])
plt.figure(figsize=(4,6))
plt.plot(gt['locx'], gt['locy'], '-', color='grey', label='Ground Truth')
plt.plot(df['pos_x'], df['pos_y'], '.-', label='Player Trace')
plt.title('Player Movement Path', fontsize=14)
#plt.scatter(beacons[:, 0], beacons[:, 1], c='red', marker='x', s=100, label='Beacons')
plt.xticks([3,6,9], fontsize=14)
plt.yticks([6,12], fontsize=14)
plt.xlim(0,12)
plt.xlabel('X (m)', fontsize=14)
plt.ylabel('Y (m)', fontsize=14)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.legend(loc='upper right', fontsize=14)
plt.tight_layout()
plt.show()