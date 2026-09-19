import pandas as pd
import joblib
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/train_features.csv"
MODEL_PATH = "models/random_forest_sales_forecaster.joblib"
OUTPUT_PATH = "data/processed/validation_forecast.csv"


def create_validation_forecast():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    validation = df[
        (df["date"] >= "2017-07-01") &
        (df["date"] <= "2017-08-15")
    ].copy()

    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    features = model_data["features"]

    validation = validation.dropna(subset=features).copy()

    validation["predicted_sales"] = model.predict(
        validation[features]
    )

    validation.to_csv(OUTPUT_PATH, index=False)

    daily = (
        validation
        .groupby("date")[["sales", "predicted_sales"]]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(14, 6))

    plt.plot(
        daily["date"],
        daily["sales"],
        label="Actual Sales"
    )

    plt.plot(
        daily["date"],
        daily["predicted_sales"],
        label="Predicted Sales"
    )

    plt.title("Actual vs Predicted Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        "data/processed/actual_vs_predicted.png",
        dpi=150
    )

    plt.show()

    print("Validation forecast created successfully.")
    print("Rows:", len(validation))
    print("Saved data:", OUTPUT_PATH)
    print("Saved chart: data/processed/actual_vs_predicted.png")


if __name__ == "__main__":
    create_validation_forecast()