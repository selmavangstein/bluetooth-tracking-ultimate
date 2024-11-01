

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os


def clean_data(filename:str):
    """Creates a csv file from our tio output txt file

    Args:
        filename (str): name of the txt file to clean

    Returns:
        str: filename of the cleaned csv file Ex. "filename_cleaned.csv"
    """

    # Convert .log file to .txt if necessary
    # if filename.endswith('.log'):
    #     new_filename = filename.replace('.log', '.txt')
    #     with open(filename, 'r') as log_file:
    #         with open(new_filename, 'w') as txt_file:
    #             txt_file.write(log_file.read())
    #     filename = new_filename

    # take a txt file, cleans it, returns a csv file
    with open(filename, 'r') as f:
        data = f.read().splitlines()

    # track whenn presses happen
    presses = [d.split("]")[0].replace("[","") for d in data if 'PRESS' in d]
    print(presses)


    # remove blank lines from file
    data = [d for d in data if d]

    # remove incomplete lines from file
    data = [d for d in data if len(d.split(",")) == 3]

    # remove spaces from file
    data = [d.replace(" ", "") for d in data]

    final_timestamps = []
    final_values = []
    for d in data:
        timestamp = d.split("]")[0].replace('[', "")
        data = d.split("]")[1]
        if "," in data:
            final_timestamps.append(timestamp)
            final_values.append(data)

    # turn into csv
    with open(f'{filename.replace(".txt", "").replace(".log", "")}_cleaned.csv', 'w') as c:
        c.write('timestamp,distance,rtt,rssi\n')
        for i in range(len(final_timestamps)):
            dist = final_values[i].split(",")[0]
            rtt = final_values[i].split(",")[1]
            rssi = final_values[i].split(",")[2]
            c.write(f'{final_timestamps[i]},{float(dist)},{float(rtt)},{float(rssi)}\n')

    return f'{filename.replace(".txt", "").replace(".log", "")}_cleaned.csv', presses


def plot_smoothed_data(filename:str, windows:list, param:str='rssi', plot:bool=True):
    # read the csv file using csv library
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # parse the csv file
    timestamps = [d[0] for d in data[1:]]
    distances = [float(d[1]) for d in data[1:]]
    rtts = [float(d[2]) for d in data[1:]]
    rssis = [float(d[3]) for d in data[1:]]

    paramchecker = {'distance': distances, 'rtt': rtts, 'rssi': rssis}

    # read the param into a pandas Series
    param_series = pd.Series(paramchecker[param])

    # get standard deviation of the param
    std = param_series.std()

    # remove outliers
    rm_param_series = []
    for v in param_series:
        if abs(v - param_series.mean()) > 2 * std:
            if v > param_series.mean():
                rm_param_series.append(v - 2*std)
            else:
                rm_param_series.append(v + 2*std)
        else:
            rm_param_series.append(v)

    rm_param_series = pd.Series(rm_param_series)

    # Calculate SMAs
    smas = []
    emas = []
    outs = [] # ema on outliers removed
    for window in windows:
        sma = []
        ema = []
        out = []
        for i in range(window-1):
            sma.append(None)

        smaz = param_series.rolling(window=window).mean().dropna().tolist()
        emaz = param_series.ewm(span=window, adjust=False).mean().dropna().tolist()
        outz = rm_param_series.ewm(span=window, adjust=False).mean().dropna().tolist()

        sma.extend(smaz)
        ema.extend(emaz)
        out.extend(outz)
        
        smas.append(sma)
        emas.append(ema)
        outs.append(out)

    # print(len(smas[0]), "\n\n\n", len(emas[0]), "\n\n\n", len(outs[0]), "\n\n\n", len(timestamps))

    # Plotting
    plt.figure(figsize=(12, 8))
    i = 0
    for sma in smas:
        # offset the sma by the window size
        plt.plot(range(len(timestamps)), sma, label=f'SMA {windows[i]}')
        i += 1

    e = 0
    for ema in emas:
        plt.plot(range(len(timestamps)), ema, label=f'EMA {windows[e]}')
        e += 1

    o = 0
    for out in outs:
        plt.plot(range(len(timestamps)), out, label=f'Outlier EMA {windows[o]}', linestyle='--') 
        o += 1

    plt.plot(range(len(timestamps)), paramchecker[param], label=param, color='black', alpha=0.5)
    plt.xlabel('Timestamp')
    plt.ylabel(param.capitalize())
    plt.title(f'{param.capitalize()} with SMAs | {filename.split("/")[-1].replace(".csv", "")}')
    plt.legend()
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig(f'{filename.replace(".csv", "")}_{param}.png')

    return smas, timestamps

    

# def smooth_data(filename:str, windows:list, param:str='rssi', plot:bool=False):
    with open(filename, 'r') as f:
        data = csv.reader(f)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename, header=None, names=['timestamp', 'distance', 'rtt', 'rssi'])

    # Extract the columns into Series
    timestamps = df['timestamp'].to_list()[1:]
    distances_series = df['distance']
    rtts_series = df['rtt']
    rssi_series = df['rssi']

    paramchecker = {'distance': distances_series[1:], 'rtt': rtts_series[1:], 'rssi': rssi_series[1:]}
    
    smas = []
    for window in windows:
        sma = paramchecker[param].rolling(window=window).mean().dropna().tolist()
        smas.append(sma)

    paramchecker = {'distance': distances_series[1:].to_list(), 'rtt': rtts_series[1:].to_list(), 'rssi': rssi_series[1:].to_list()}

    # Plotting
    if plot:
        i = 0
        for sma in smas:
            plt.plot(range(len(timestamps[windows[i]-1:])), sma, label=f'SMA {windows[i]}')
            i += 1

        plt.plot(range(len(timestamps)), paramchecker[param], label=param, color='black', alpha=0.5)
        plt.tight_layout()
        plt.legend()
        plt.show()

    return smas, timestamps




directory = '/Users/cullenbaker/comps/bluetooth-tracking-ultimate/makecharts'
for filename in os.listdir(directory):
    if filename.endswith('.log') or filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        cleaned_filename, presses = clean_data(filepath)
        plot_smoothed_data(cleaned_filename, [100, 200], param='distance', plot=False)


# Example usage
# filename = "20m obstacle test.log"
# clean_data(filename)
# plot_smoothed_data(f'{filename.replace(".txt", "").replace(".log", "")}_cleaned.csv', [100, 150], param='rtt')
# smooth_data('stationary1_cleaned.csv', [50, 100, 150], plot=True)

"""
with open('30ydstop.txt') as f:
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


exit()

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

"""

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
