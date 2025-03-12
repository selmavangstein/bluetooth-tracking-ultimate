import pandas as pd
import re
from matplotlib import pyplot as plt

def plot_data(df_curr: pd.DataFrame, df_prev: pd.DataFrame, gt=None, title=""):
    """
    Analyzes measured vs. ground truth data when:
      - df_meas is in centimeters, so we scale by /100
      - df_gt is already in meters, so we do NOT rescale it

    Steps:
      1) Parse timestamps with format="%H:%M:%S.%f".
      2) Identify beacon columns:
         - For measured, scale columns by /100 if they are in cm.
         - For ground truth, do NOT scale, since it's already in m.
      3) Sort & merge left=GT, right=Meas, with 1s tolerance.
      4) Compute Error => measured - ground_truth => bXd_y - bXd_x
      5) Plot:
         - red line => all ground truth rows
         - black line => all measured rows
         - green x => merged measured points
      6) Save merged to "merged_results.csv"
    """
    df_curr = df_curr.copy()

    # 1) Parse timestamps with "HH:MM:SS.sss" if needed
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

    # 2) Identify columns
    meas_beacon_cols = [c for c in df_curr.columns if re.match(r'^b\d+d$', c)]

    # 5) Plot lines
    fig, ax = plt.subplots(nrows=1, ncols=1,
                             figsize=(10, len(meas_beacon_cols)), sharex=True)
    if len(meas_beacon_cols) == 1:
        axes = [axes]


    # Plot prev from df_gt_sorted
    ax.plot(
        gt['timestamp'], gt['b1d'],
        '--.', color="dimgrey", label=f"Ground Truth"
    )

    # Plot prev from df_gt_sorted
    ax.plot(
        df_prev['timestamp'], df_prev['b1d'],
        'r.-', label=f"Prev Dist"
    )
    #Plot curr from df_meas_sorted
    ax.plot(
        df_curr['timestamp'], df_curr['b1d'],
        'k.-', label=f"Curr Dist"
    )
    # Plot merged measured from df_merged
    # meas_col_merged = c + "_y"
    # if meas_col_merged in df_merged.columns:
    #     ax.plot(
    #         df_merged['timestamp'], df_merged[meas_col_merged],
    #         'gx', label=f"Merged Dist{anchor_id}"
    #     )

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