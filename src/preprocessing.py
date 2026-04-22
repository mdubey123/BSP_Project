import pandas as pd

def clean_data(df):
    # Drop columns with >80% missing values
    threshold = 0.8
    df = df.loc[:, df.isnull().mean() < threshold]

    # Drop all-zero columns
    zero_cols = [col for col in df.columns if (df[col] == 0).all()]
    df = df.drop(columns=zero_cols)

    # Convert important numeric columns
    numeric_cols = [
        "PR_VALUE", "NEGOTIATION_VAL", "PO_VALUE", "L1_VALUE",
        "RESPONSE", "NO_OF_TECHSUIT", "NO_OF_EXT",
        "DUR_OF_CONTRACT", "PER_COMPLETED", "EXECUTED_QTY",
        "DURATION_LEFT"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove unusable rows for modelling
    required_cols = [c for c in ["PR_VALUE", "NEGOTIATION_VAL"] if c in df.columns]
    if required_cols:
        df = df.dropna(subset=required_cols)

    if "PR_VALUE" in df.columns:
        df = df[df["PR_VALUE"] > 0]

    if "NEGOTIATION_VAL" in df.columns:
        df = df[df["NEGOTIATION_VAL"] >= 0]
    return df


def handle_missing(df):
    missing = df.isnull().mean() * 100
    return missing.sort_values(ascending=False)