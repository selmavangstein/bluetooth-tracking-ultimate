import pandas as pd


input_file = "yourfile.csv"
output_file = "cleaned_file.csv"

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

