import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Saving Distribution Plot
# -----------------------------
def plot_saving_distribution(df):
    # Remove extreme outliers (1% - 99%)
    lower = df['saving'].quantile(0.01)
    upper = df['saving'].quantile(0.99)

    df_filtered = df[(df['saving'] >= lower) & (df['saving'] <= upper)].copy()

    plt.figure(figsize=(10, 6))
    sns.histplot(df_filtered['saving'], bins=50, kde=True)

    plt.title("Filtered Saving Distribution")
    plt.xlabel("Saving")
    plt.ylabel("Frequency")

    plt.show()


# -----------------------------
# Vendor Analysis (FIXED)
# -----------------------------
def vendor_analysis(df):
    # Remove missing vendor names
    df = df.dropna(subset=['L1_PARTY_NAME'])

    # Remove blank / empty vendor names
    df = df[df['L1_PARTY_NAME'].str.strip() != ""]

    # Group by vendor and calculate total savings
    vendor_savings = df.groupby('L1_PARTY_NAME')['saving'].sum()

    # Sort and return top 10 vendors
    return vendor_savings.sort_values(ascending=False).head(10)