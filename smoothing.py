

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

with open('log2.txt') as f:
    data = f.read().splitlines()
    for d in data:
        print(d)

    timestamps = [d.split()[0].replace('[', "").replace(']', "") for d in data]
    rssi = [float(d.split()[1]) for d in data]

    # Create a pandas Series from the rssi list
    rssi_series = pd.Series(rssi)

    # Calculate the simple moving average with a window size of 3
    # simple_moving_avg_ten = rssi_series.rolling(window=10).mean().dropna().tolist()
    # simple_moving_avg_3 = rssi_series.rolling(window=3).mean().dropna().tolist()
    # simple_moving_avg_15 = rssi_series.rolling(window=15).mean().dropna().tolist()
    simple_moving_avg_50 = rssi_series.rolling(window=50).mean().dropna().tolist()
    simple_moving_avg_100 = rssi_series.rolling(window=100).mean().dropna().tolist()
    simple_moving_avg_150 = rssi_series.rolling(window=150).mean().dropna().tolist()

    print(simple_moving_avg_50)
    print(simple_moving_avg_100)
    print(simple_moving_avg_150)

    # Plotting
    # plt.plot(range(len(timestamps))[9:], simple_moving_avg_ten, label='SMA 10')
    # plt.plot(range(len(timestamps))[2:], simple_moving_avg_3, label='SMA 3')
    # plt.plot(range(len(timestamps))[14:], simple_moving_avg_15, label='SMA 15')
    plt.plot(range(len(timestamps)), rssi, label='RSSI', color='black', alpha=0.5)
    plt.plot(range(len(timestamps))[49:], simple_moving_avg_50, label='SMA 50')
    plt.plot(range(len(timestamps))[99:], simple_moving_avg_100, label='SMA 100')
    plt.plot(range(len(timestamps))[149:], simple_moving_avg_150, label='SMA 150')
    
    plt.tight_layout()
    plt.legend()
    plt.show()

# do this on csv data
with open('walking_ble.csv', 'r') as c:
    csv_reader = csv.reader(c)
    data = []
    for row in csv_reader:
        data.append(row)

    timestamps = [d[0] for d in data]
    rssi = [float(d[1]) for d in data]

    # Create a pandas Series from the rssi list
    rssi_series = pd.Series(rssi)

    # Calculate the simple moving average with a window size of 3
    simple_moving_avg_5 = rssi_series.rolling(window=5).mean().dropna().tolist()
    simple_moving_avg_50 = rssi_series.rolling(window=15).mean().dropna().tolist()
    simple_moving_avg_100 = rssi_series.rolling(window=20).mean().dropna().tolist()
    simple_moving_avg_150 = rssi_series.rolling(window=30).mean().dropna().tolist()

    print(simple_moving_avg_5)
    print(simple_moving_avg_50)
    print(simple_moving_avg_100)
    print(simple_moving_avg_150)

    # Plotting
    plt.plot(range(len(timestamps)), rssi, label='RSSI', color='black', alpha=0.5)
    plt.plot(range(len(timestamps))[4:], simple_moving_avg_5, label='SMA 5')
    plt.plot(range(len(timestamps))[14:], simple_moving_avg_50, label='SMA 50')
    plt.plot(range(len(timestamps))[19:], simple_moving_avg_100, label='SMA 100')
    plt.plot(range(len(timestamps))[29:], simple_moving_avg_150, label='SMA 150')
    
    plt.tight_layout()
    plt.legend()
    plt.show()



# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# with open('log2.txt') as f:
#     data = f.read().splitlines()
#     for d in data:
#         print(d)

#     timestamps = [d.split()[0].replace('[', "").replace(']', "") for d in data]
#     rssi = [float(d.split()[1]) for d in data]

#     # Create a pandas Series from the rssi list
#     rssi_series = pd.Series(rssi)

#     # Calculate the exponential moving average with different spans
#     ema_10 = rssi_series.ewm(span=10, adjust=False).mean().tolist()
#     ema_3 = rssi_series.ewm(span=3, adjust=False).mean().tolist()
#     ema_15 = rssi_series.ewm(span=15, adjust=False).mean().tolist()
#     ema_20 = rssi_series.ewm(span=20, adjust=False).mean().tolist()
#     ema_50 = rssi_series.ewm(span=50, adjust=False).mean().tolist()

#     # Plotting
#     plt.plot(range(len(timestamps)), ema_10, label='EMA 10')
#     plt.plot(range(len(timestamps)), ema_3, label='EMA 3')
#     plt.plot(range(len(timestamps)), ema_15, label='EMA 15')
#     plt.plot(range(len(timestamps)), ema_20, label='EMA 20')
#     plt.plot(range(len(timestamps)), ema_50, label='EMA 50')
#     plt.plot(range(len(timestamps)), rssi, label='RSSI')
#     plt.legend()
#     plt.show()
