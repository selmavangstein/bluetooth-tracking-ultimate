import pandas as pd
import re
from matplotlib import pyplot as plt

def plot_data(df_curr: pd.DataFrame, df_prev: pd.DataFrame, gt=None, title=""):
    """Plot two steps from the pipeline with ground truth on the same plot
    Plotting methods borrowed from analyze_ftm_data"""
    df_curr = df_curr.copy()

   # For df_curr:
    if not pd.api.types.is_datetime64_any_dtype(df_curr['timestamp']):
        df_curr['timestamp'] = pd.to_datetime(df_curr['timestamp'], errors='raise') \
                                .apply(lambda t: t.replace(year=1900, month=1, day=1))

    # For df_prev:
    if not pd.api.types.is_datetime64_any_dtype(df_prev['timestamp']):
        df_prev['timestamp'] = pd.to_datetime(df_prev['timestamp'], errors='raise') \
                                .apply(lambda t: t.replace(year=1900, month=1, day=1))
        
    # For gt:
    if not pd.api.types.is_datetime64_any_dtype(gt['timestamp']):
        gt['timestamp'] = pd.to_datetime(gt['timestamp'], errors='raise') \
                                .apply(lambda t: t.replace(year=1900, month=1, day=1))

    meas_beacon_cols = [c for c in df_curr.columns if re.match(r'^b\d+d$', c)]

    fig, ax = plt.subplots(nrows=1, ncols=1,
                             figsize=(10, len(meas_beacon_cols)), sharex=True)
    if len(meas_beacon_cols) == 1:
        axes = [axes]

    # Plot ground truth
    ax.plot(
        gt['timestamp'], gt['b1d'],
        '--.', color="dimgrey", label=f"Ground Truth"
    )

    # Plot prev
    ax.plot(
        df_prev['timestamp'], df_prev['b1d'],
        'r.-', label=f"Prev Dist"
    )
    #Plot curr
    ax.plot(
        df_curr['timestamp'], df_curr['b1d'],
        'k.-', label=f"Curr Dist"
    )

    ax.set_title(title)
    ax.set_ylabel("Distance (m)")
    ax.set_ylim(0, 50)
    ax.legend()
    ax.grid(True)

    ax.set_xlabel("Time")
    plt.tight_layout()
    plt.show()

curr = pd.read_csv("processedKalman Filter3.csv")
prev = pd.read_csv("processedVelocity Clamping2.csv")
gt = pd.read_csv('data/GT-obstacletest2.csv')

plot_data(curr,prev,gt=gt, title="Distance Correction")