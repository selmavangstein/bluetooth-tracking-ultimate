
"""
Cleans the raw output from the beacon and converts it into a correctly formatted csv
"""

def clean(logFilePath, newFileNamePath):
    """
    Cleans log file wil format like so:
    [16:36:23.182] 1350063,23.11,21.66,22.91,24.13,-0.51,10.04,5.33
    turns into csv
    """

    with open(logFilePath, 'r') as f:
        lines = f.readlines()

    with open(newFileNamePath, 'w') as f:
        f.write("timestamp,wearabletimestamp,b1d,b2d,b3d,b4d,xa,ya,za\n")
        for line in lines:
            # if line is not in lof format, skip
            if not line.startswith("["):
                continue
            line = line.replace("[", "")
            line = line.replace("] ", ",")
            line = line.replace(" ", "")
            f.write(line)

if __name__ == "__main__":
    log = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/data/rawdataexample.log"
    new = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/data/rawdataexample.csv"
    clean(log, new)
