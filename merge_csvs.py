import csv
from datetime import datetime

def merge_csvs(ground_truth_path, velocity_clamp_path, output_path):
    # Read ground truth data
    with open(ground_truth_path, 'r') as gt_file:
        gt_reader = csv.reader(gt_file)
        gt_header = next(gt_reader)
        gt_header = ['gt_' + col for col in gt_header]  # Change ground truth column names
        gt_data = {row[0]: row[1:] for row in gt_reader}

    # Read velocity clamp data
    with open(velocity_clamp_path, 'r') as vc_file:
        vc_reader = csv.reader(vc_file)
        vc_header = next(vc_reader)
        vc_data = {}
        for row in vc_reader:
            timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%d %H:%M:%S")
            if timestamp not in vc_data:
                vc_data[timestamp] = []
            vc_data[timestamp].append(row[1:])

    # Merge data
    merged_data = []
    for timestamp, gt_row in gt_data.items():
        if timestamp in vc_data:
            for vc_row in vc_data[timestamp]:
                merged_data.append([timestamp] + gt_row + vc_row)

    # Write merged data to output CSV
    with open(output_path, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(['timestamp'] + gt_header[1:] + vc_header[1:])
        writer.writerows(merged_data)

# Example usage
merge_csvs(
    '/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/player_positions_Ground Truth.csv',
    '/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/player_positions_Velocity Clamping5.csv',
    '/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/merged_player_positions.csv'
)
