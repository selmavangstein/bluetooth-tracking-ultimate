import pandas as pd

# Load CSV file
df = pd.read_csv("data/standing still.csv")

print("TESTING VARIANCE")
# Compute variance of the 'distance' column
variance1 = df['b1d'].var()
print("Variance 1:", variance1)

variance2 = df['b2d'].var()
print("Variance 2:", variance2)

variance3 = df['b3d'].var()
print("Variance 3:", variance3)

variance4 = df['b4d'].var()
print("Variance 4:", variance4)

print("ave var: ", (variance1+variance2+variance3)/3) #ignoring 4, had an obstacle



#TIME FLUCTUATION CHECK
print("dt FLUCTUATION CHECK")
# Load CSV file
df = pd.read_csv("data/standing still.csv", parse_dates=['timestamp'])  # Ensure timestamps are datetime

# Compute time differences in seconds
time_diffs = df['timestamp'].diff().dt.total_seconds()

# Compute the average time difference
avg_diff = time_diffs.mean()

# Count how many rows exceed 10% fluctuation
num_exceeded = (time_diffs > (1.05 * avg_diff)).sum()

# Print the result
print(f"Number of rows with time differences exceeding 10%: {num_exceeded}")


print("SYSTEM ERROR CHECK")
mean1 = df['b1d'].mean()
print("Error 1:", mean1-10.8)

mean2 = df['b2d'].mean()
print("Error 2:", mean2-10.8)

mean3 = df['b3d'].mean()
print("Error 3:", mean3-10.8)