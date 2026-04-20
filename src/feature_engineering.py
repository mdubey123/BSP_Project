import pandas as pd

def create_features(df):
    # Convert numeric columns safely
    num_cols = ['PR_VALUE', 'PO_VALUE', 'L1_VALUE', 'NEGOTIATION_VAL']

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create saving
    df['saving'] = df['PR_VALUE'] - df['PO_VALUE']

    # Negotiation impact
    if 'L1_VALUE' in df.columns:
        df['negotiation_impact'] = df['L1_VALUE'] - df['PO_VALUE']

    # Saving percentage (avoid division error)
    df['saving_percent'] = (df['saving'] / df['PR_VALUE'].replace(0, pd.NA)) * 100

    return df