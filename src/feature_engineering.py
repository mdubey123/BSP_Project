import pandas as pd

def create_features(df):
    # Convert numeric columns safely
    num_cols = ['PR_VALUE', 'PO_VALUE', 'L1_VALUE', 'NEGOTIATION_VAL']

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # IMPORTANT:
    # For the current prediction UI, saving must be based on
    # PR_VALUE and NEGOTIATION_VAL, because those are the inputs
    # given to the model in the frontend.
    df['saving'] = df['PR_VALUE'] - df['NEGOTIATION_VAL']

    # Optional supporting feature
    if 'L1_VALUE' in df.columns:
        df['negotiation_impact'] = df['L1_VALUE'] - df['NEGOTIATION_VAL']

    # Saving percentage
    df['saving_percent'] = (df['saving'] / df['PR_VALUE'].replace(0, pd.NA)) * 100

    return df