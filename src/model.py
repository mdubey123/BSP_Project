from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(df):
    # DO NOT include PR_VALUE and PO_VALUE together
    df_ml = df[['PR_VALUE', 'NEGOTIATION_VAL', 'saving']].dropna()

    # Features & target
    X = df_ml[['PR_VALUE', 'NEGOTIATION_VAL']]
    y = df_ml['saving']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"mae": mae, "r2": r2}