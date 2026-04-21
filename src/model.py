from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(df):
    df_ml = df[['PR_VALUE', 'NEGOTIATION_VAL', 'saving']].dropna()

    X = df_ml[['PR_VALUE', 'NEGOTIATION_VAL']]
    y = df_ml['saving']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance:")
    print("MAE:", mae)
    print("R2:", r2)

    return model