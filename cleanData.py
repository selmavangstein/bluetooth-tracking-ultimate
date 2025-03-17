""" 
functions for cleaning raw data and processing a clean csv
"""
import pandas as pd
import os

def processData(filename, tests):
    # Load initial DF
    initalDf = loadData(os.path.join(os.getcwd(), filename))
    # Check if the first row has the required headers
    required_headers = ["timestamp", "wearabletimestamp", "b1d", "b2d", "b3d", "b4d", "xa", "ya", "za"]
    if list(initalDf.columns) != required_headers:
        # Add the required headers
        initalDf.columns = required_headers

    #remove error messages from data
    initalDf = cleanup_file(initalDf)


    # Ensure all values are floats/ints and not strings
    for column in initalDf.columns:
        if initalDf[column].dtype == 'object':
            try:
                initalDf[column] = initalDf[column].astype(float)
            except ValueError:
                print(f"Column {column} cannot be converted to float.")
    dfs = [initalDf]

    # Run Tests on DF
    for testname, test in tests:
        df = dfs[-1]
        resultingDF = test(df)
        print(f"Test {testname} complete")
        # Append the resulting DF to the list of data
        dfs.append(resultingDF)

    # Save all the DFS
    final = []
    final.append(("Initial", initalDf))
    i = 0
    for d in dfs[1:]:
        final.append((tests[i][0] + str(i), d))
        i += 1

    # Return a list of all the dataframes we created, final df is [-1]
    return final

def is_valid_row(row):
    """Checks if a row follows the expected numeric format."""
    try:
        values = row.split(',')
        if len(values) != 9:
            return False
        for val in values[1:]:
            float(val)  # Try converting to float
        return True
    except ValueError:
        return False  # If conversion fails, it's an invalid row

def cleanup_file(df):
    df_cleaned = df[df.apply(lambda row: is_valid_row(','.join(row.astype(str))), axis=1)].copy()
    df_cleaned.reset_index(drop=True, inplace=True)
    return df_cleaned

def loadData(filename):
    """
    Loads the data from a csv file into a pandas dataframe
    """
    return pd.read_csv(filename)

def clean(logFilePath, newFileNamePath):
    """
    Cleans log file wil format like so:
    [16:36:23.182],playerid,1350063,23.11,21.66,22.91,24.13,-0.51,10.04,5.33
    turns into csv
    """

    with open(logFilePath, 'r') as f:
        lines = f.readlines()

    lines = lines[1:]

    with open(newFileNamePath, 'w') as f:
        f.write("timestamp,playerid,wearabletimestamp,b1d,b2d,b3d,b4d,xa,ya,za\n")
        for line in lines:
            # if line is not in lof format, skip
            if not line.startswith("["):
                continue
            line = line.replace("[", "")
            line = line.replace("] ", ",")
            line = line.replace(" ", "")
            f.write(line)

    return newFileNamePath

if __name__ == "__main__":
    log = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/data/rawdataexample.log"
    new = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/data/rawdataexample.csv"
    clean(log, new)
