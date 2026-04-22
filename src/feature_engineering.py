import pandas as pd

def create_features(df):
    # Convert numeric columns safely
    num_cols = [
        "PR_VALUE", "PO_VALUE", "L1_VALUE", "NEGOTIATION_VAL",
        "RESPONSE", "NO_OF_TECHSUIT", "NO_OF_EXT",
        "DUR_OF_CONTRACT", "PER_COMPLETED", "EXECUTED_QTY",
        "DURATION_LEFT"
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Business target
    if "PR_VALUE" in df.columns and "NEGOTIATION_VAL" in df.columns:
        df["saving"] = df["PR_VALUE"] - df["NEGOTIATION_VAL"]
        df["saving_percent"] = (df["saving"] / df["PR_VALUE"].replace(0, pd.NA)) * 100
    else:
        raise ValueError("Dataset must contain PR_VALUE and NEGOTIATION_VAL columns.")
    # Negotiation impact
    if "L1_VALUE" in df.columns:
        df["negotiation_impact"] = df["L1_VALUE"] - df["NEGOTIATION_VAL"]

    # Date-derived features
    date_candidates = ["REQUEST_DATE", "ENQUIRY_DATE", "APPROVAL_DATE", "INSERT_DATE"]
    for col in date_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "REQUEST_DATE" in df.columns:
        df["request_month"] = df["REQUEST_DATE"].dt.month
        df["request_quarter"] = df["REQUEST_DATE"].dt.quarter
    elif "INSERT_DATE" in df.columns:
        df["request_month"] = df["INSERT_DATE"].dt.month
        df["request_quarter"] = df["INSERT_DATE"].dt.quarter

    # Classification target
    def classify_saving_pct(pct):
        if pd.isna(pct):
            return None
        if pct < 0:
            return "Loss"
        elif pct <= 10:
            return "Low Saving"
        else:
            return "High Saving"

    df["saving_class"] = df["saving_percent"].apply(classify_saving_pct)
   

    return df