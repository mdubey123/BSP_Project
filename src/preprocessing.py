import pandas as pd

def clean_data(df):
    # Drop columns with >80% missing
    threshold = 0.8
    df = df.loc[:, df.isnull().mean() < threshold]

    # Drop all-zero columns
    zero_cols = [col for col in df.columns if (df[col] == 0).all()]
    df = df.drop(columns=zero_cols)

    return df


def handle_missing(df):
    # Show percentage of missing values
    missing = df.isnull().mean() * 100
    return missing.sort_values(ascending=False)