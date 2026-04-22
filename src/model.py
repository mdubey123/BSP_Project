from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report

def train_model(df):
    numeric_features = [
        "PR_VALUE",
        "RESPONSE",
        "NO_OF_TECHSUIT",
        "NO_OF_EXT",
        "DUR_OF_CONTRACT",
        "request_month",
        "request_quarter"
    ]

    categorical_features = [
        "RA_IND",
        "MSME_DET",
        "WERKS",
        "PUR_GROUP"
    ]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]
    all_features = numeric_features + categorical_features

    def make_preprocessor():
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median"))
                    ]),
                    numeric_features
                ),
                (
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ]),
                    categorical_features
                ),
            ]
        )

    reg_preprocessor = make_preprocessor()
    cls_preprocessor = make_preprocessor()

    # Regression model
    reg_df = df[all_features + ["NEGOTIATION_VAL"]].dropna(subset=["NEGOTIATION_VAL"]).copy()
    X_reg = reg_df[all_features]
    y_reg = reg_df["NEGOTIATION_VAL"]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    reg_model = Pipeline([
        ("preprocessor", reg_preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42
        ))
    ])

    reg_model.fit(Xr_train, yr_train)
    reg_pred = reg_model.predict(Xr_test)

    print("\nRegression Model Performance:")
    print("MAE:", mean_absolute_error(yr_test, reg_pred))
    print("R2:", r2_score(yr_test, reg_pred))

    # Classification model
    cls_df = df[all_features + ["saving_class"]].dropna(subset=["saving_class"]).copy()
    X_cls = cls_df[all_features]
    y_cls = cls_df["saving_class"]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    cls_model = Pipeline([
        ("preprocessor", cls_preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            random_state=42
        ))
    ])

    cls_model.fit(Xc_train, yc_train)
    cls_pred = cls_model.predict(Xc_test)

    print("\nClassification Model Performance:")
    print("Accuracy:", accuracy_score(yc_test, cls_pred))
    print(classification_report(yc_test, cls_pred))

    # Anomaly model
    anomaly_features = [
        c for c in [
            "PR_VALUE",
            "NEGOTIATION_VAL",
            "saving",
            "saving_percent",
            "RESPONSE",
            "NO_OF_TECHSUIT",
            "NO_OF_EXT",
            "DUR_OF_CONTRACT"
        ] if c in df.columns
    ]

    anomaly_df = df[anomaly_features].copy()

    anomaly_preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    anomaly_X = anomaly_preprocessor.fit_transform(anomaly_df)

    anomaly_model = IsolationForest(
        contamination=0.05,
        random_state=42
    )
    anomaly_model.fit(anomaly_X)

    return {
        "reg_model": reg_model,
        "cls_model": cls_model,
        "anomaly_model": anomaly_model,
        "anomaly_preprocessor": anomaly_preprocessor,
        "features": all_features,
        "anomaly_features": anomaly_features
    }