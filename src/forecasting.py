import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


DATA_PATH = "data/processed/train_features.csv"
MODEL_PATH = "models/random_forest_sales_forecaster.joblib"


def train_and_save_model():

    # -----------------------------------
    # Load Data
    # -----------------------------------
    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------------
    # Training Data
    # -----------------------------------
    train = df[
        df["date"] < "2017-07-01"
    ].copy()

    # -----------------------------------
    # Features
    # -----------------------------------
    features = [
        "store_nbr",
        "onpromotion",
        "year",
        "month",
        "day",
        "day_of_week",
        "week_of_year",
        "quarter",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_7",
        "rolling_14",
        "rolling_28",
    ]

    target = "sales"

    # -----------------------------------
    # Remove missing features
    # -----------------------------------
    train = train.dropna(
        subset=features
    )

    # -----------------------------------
    # Training sample
    # -----------------------------------
    train_sample = train.sample(
        n=500_000,
        random_state=42
    )

    X_train = train_sample[features]
    y_train = train_sample[target]

    print("Training rows:", len(X_train))
    print("Features:", len(features))

    # -----------------------------------
    # Tuned Random Forest
    # -----------------------------------
    model = RandomForestRegressor(
        n_estimators=75,
        max_depth=18,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=2
    )

    print("\nTraining final Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # -----------------------------------
    # Save Model
    # -----------------------------------
    joblib.dump(
        {
            "model": model,
            "features": features
        },
        MODEL_PATH
    )

    print("\nModel saved successfully.")
    print("Path:", MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()