import pandas as pd

def load_data(path):
    # Load all sheets
    all_sheets = pd.read_excel(path, sheet_name=None)

    # Combine all sheets into one DataFrame
    df = pd.concat(all_sheets.values(), ignore_index=True)

    return df