import pandas as pd

def clean_data(df):
    # Drop columns with >80% missing values
    threshold = 0.8
    df = df.loc[:, df.isnull().mean() < threshold]

    # Drop all-zero columns
    zero_cols = [col for col in df.columns if (df[col] == 0).all()]
    df = df.drop(columns=zero_cols)

    # Convert important numeric columns safely
    numeric_cols = ["PR_VALUE", "NEGOTIATION_VAL", "PO_VALUE", "L1_VALUE"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid rows required for prediction/modeling
    required_cols = ["PR_VALUE", "NEGOTIATION_VAL"]
    existing_required = [col for col in required_cols if col in df.columns]

    if existing_required:
        df = df.dropna(subset=existing_required)

    if "PR_VALUE" in df.columns:
        df = df[df["PR_VALUE"] > 0]

    if "NEGOTIATION_VAL" in df.columns:
        df = df[df["NEGOTIATION_VAL"] > 0]

    return df


def handle_missing(df):
    # Show percentage of missing values
    missing = df.isnull().mean() * 100
    return missing.sort_values(ascending=False)