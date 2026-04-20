from src.data_loader import load_data
from src.preprocessing import clean_data, handle_missing
from src.feature_engineering import create_features
from src.analysis import plot_saving_distribution, vendor_analysis
from src.model import train_model


def main():
    # -----------------------------
    # 1. Load Dataset
    # -----------------------------
    df = load_data("data/dataset.xlsx")

    print("Dataset Loaded Successfully ✅")
    print("Shape:", df.shape)

    # -----------------------------
    # 2. Clean Data
    # -----------------------------
    df = clean_data(df)
    print("After Cleaning Shape:", df.shape)

    # -----------------------------
    # 3. Missing Values Analysis
    # -----------------------------
    missing = handle_missing(df)
    print("\nTop Missing Values:\n", missing.head(10))

    # -----------------------------
    # 4. Feature Engineering
    # -----------------------------
    df = create_features(df)

    # -----------------------------
    # 5. Visualization
    # -----------------------------
    plot_saving_distribution(df)

    # -----------------------------
    # 6. Vendor Analysis
    # -----------------------------
    top_vendors = vendor_analysis(df)
    print("\nTop Vendors:\n", top_vendors)

    # -----------------------------
    # 7. Machine Learning Model
    # -----------------------------
    results = train_model(df)

    print("\nModel Performance:")
    print("MAE:", results['mae'])
    print("R2:", results['r2'])


# -----------------------------
# Run Program
# -----------------------------
if __name__ == "__main__":
    main()