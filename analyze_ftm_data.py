# analyze_ftm_data.py
# Final - Scale only the measured columns
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_ftm_data(df_meas: pd.DataFrame, df_gt: pd.DataFrame, plot=False, title=""):
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
    df_meas = df_meas.copy()

    # 1) Parse timestamps with "HH:MM:SS.sss" if needed
    if not pd.api.types.is_datetime64_any_dtype(df_meas['timestamp']):
        df_meas['timestamp'] = pd.to_datetime(
            df_meas['timestamp'], format="%H:%M:%S.%f", errors='raise'
        )
    if not pd.api.types.is_datetime64_any_dtype(df_gt['timestamp']):
        df_gt['timestamp']   = pd.to_datetime(
            df_gt['timestamp'], format="%H:%M:%S.%f", errors='raise'
        )

    # 2) Identify columns
    meas_beacon_cols = [c for c in df_meas.columns if re.match(r'^b\d+d$', c)]

    # Scale only the measured columns from cm->m
    #for c in meas_beacon_cols:
    #    df_meas[c] /= 100.0

    # NO scaling for ground truth, assuming already in meters

    # 3) Sort & merge on nearest timestamp with 1s tolerance
    df_meas_sorted = df_meas.sort_values('timestamp').reset_index(drop=True)
    df_gt_sorted   = df_gt.sort_values('timestamp').reset_index(drop=True)

    df_merged = pd.merge_asof(
        df_gt_sorted,
        df_meas_sorted,
        left_on='timestamp',
        right_on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta("1s")
    )

    # 4) Compute error => measured - ground_truth => bXd_y - bXd_x
    error_cols = []
    for c in meas_beacon_cols:
        anchor_ids = re.findall(r'\d+', c)
        if not anchor_ids:
            continue
        anchor_id = anchor_ids[0]

        gt_col   = c + "_x"  # from left side (GT)
        meas_col = c + "_y"  # from right side (Measured)
        if gt_col not in df_merged.columns or meas_col not in df_merged.columns:
            print(f"Warning: no matched columns for '{c}' => skipping error calc.")
            continue

        err_col = f"Error{anchor_id}"
        df_merged[err_col] = df_merged[meas_col] - df_merged[gt_col]
        error_cols.append(err_col)

    # Print error stats
    for err_col in error_cols:
        anchor_id = re.findall(r'\d+', err_col)[0]
        mae  = df_merged[err_col].abs().mean()
        rmse = np.sqrt((df_merged[err_col]**2).mean())
        print(f"Anchor {anchor_id}: MAE={mae:.3f} m, RMSE={rmse:.3f} m")

    # 5) Plot lines
    fig, axes = plt.subplots(nrows=len(meas_beacon_cols), ncols=1,
                             figsize=(10, 4*len(meas_beacon_cols)), sharex=True)
    if len(meas_beacon_cols) == 1:
        axes = [axes]

    for ax, c in zip(axes, meas_beacon_cols):
        anchor_ids = re.findall(r'\d+', c)
        if not anchor_ids:
            continue
        anchor_id = anchor_ids[0]

        # Plot ground truth from df_gt_sorted
        ax.plot(
            df_gt_sorted['timestamp'], df_gt_sorted[c],
            'ro-', label=f"GT Dist{anchor_id}"
        )
        # Plot measured from df_meas_sorted
        ax.plot(
            df_meas_sorted['timestamp'], df_meas_sorted[c],
            'k.-', label=f"Measured Dist{anchor_id}"
        )
        # Plot merged measured from df_merged
        meas_col_merged = c + "_y"
        if meas_col_merged in df_merged.columns:
            ax.plot(
                df_merged['timestamp'], df_merged[meas_col_merged],
                'gx', label=f"Merged Dist{anchor_id}"
            )

        ax.set_title(f"Anchor {anchor_id} - Measured vs GT")
        ax.set_ylabel("Distance (m)")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), f'charts/{title}_gt.png'))
    if plot: plt.show()
    plt.close()

    # 6) Save
    df_merged.to_csv("merged_results.csv", index=False)
    # print("Done! Results saved to 'merged_results.csv'.")

    return df_merged


if __name__ == "__main__":
    # Example usage

    df_meas_example = pd.DataFrame({
        "timestamp": ["11:19:43.179","11:19:50.179","11:20:55.179"],
        "b1d": [885, 900, 3640],     # cm => 8.85, 9.00, 36.40
        "b2d": [4350,1155,1120],     # cm => 43.50, 11.55, 11.20
    })
    df_gt_example = pd.DataFrame({
        "timestamp": ["11:19:50.179","11:20:55.179"],
        "b1d": [10,36.4],            # *in meters*
        "b2d": [50,11.2],            # also in meters? Then 50 would be 50.0 m
    })
    analyze_ftm_data(df_meas_example, df_gt_example)



# def main():
#     # 1) Run the pipeline
#     dfs = processData()  # Suppose it returns a list of (title, df) tuples

    
#     # 2) Load the single ground-truth CSV
    
#     df_gt = pd.read_csv("GroundyTruthy.csv")

    
#     beaconPositions = np.array([
#         [0,  0],   
#         [20,   0],   
#         [0, 40]
#     ])
#     for title, df in dfs:
#         plotPlayers((title, df), beaconPositions)

    
#     # 4) Compare each pipeline DF to ground truth
    
#     for title, df_meas in dfs:
#         print(f"\nAnalyzing pipeline output '{title}' against ground truth...")
#         df_merged = analyze_ftm_data(df_meas, df_gt)

#     print("\nAll pipeline DataFrames processed and compared to ground truth.")
