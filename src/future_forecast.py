import pandas as pd
import joblib


TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
MODEL_PATH = "models/random_forest_sales_forecaster.joblib"
OUTPUT_PATH = "data/processed/future_forecast.csv"


def create_future_forecast():
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=["date", "store_nbr", "family", "sales", "onpromotion"]
    )

    test = pd.read_csv(
        TEST_PATH,
        usecols=["id", "date", "store_nbr", "family", "onpromotion"]
    )

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    features = model_data["features"]

    history = train[
        ["date", "store_nbr", "family", "sales", "onpromotion"]
    ].copy()

    future_rows = []

    for date in sorted(test["date"].unique()):

        current = test[test["date"] == date].copy()

        combined = pd.concat(
            [history, current.assign(sales=pd.NA)],
            ignore_index=True
        )

        combined = combined.sort_values(
            ["store_nbr", "family", "date"]
        ).reset_index(drop=True)

        combined["year"] = combined["date"].dt.year
        combined["month"] = combined["date"].dt.month
        combined["day"] = combined["date"].dt.day
        combined["day_of_week"] = combined["date"].dt.dayofweek
        combined["week_of_year"] = (
            combined["date"].dt.isocalendar().week.astype(int)
        )
        combined["quarter"] = combined["date"].dt.quarter
        combined["is_weekend"] = (
            combined["day_of_week"] >= 5
        ).astype(int)

        group = combined.groupby(
            ["store_nbr", "family"],
            sort=False
        )["sales"]

        combined["lag_1"] = group.shift(1)
        combined["lag_7"] = group.shift(7)
        combined["lag_14"] = group.shift(14)
        combined["lag_28"] = group.shift(28)

        combined["rolling_7"] = (
            combined.groupby(["store_nbr", "family"])["sales"]
            .transform(
                lambda x: x.shift(1).rolling(7).mean()
            )
        )

        combined["rolling_14"] = (
            combined.groupby(["store_nbr", "family"])["sales"]
            .transform(
                lambda x: x.shift(1).rolling(14).mean()
            )
        )

        combined["rolling_28"] = (
            combined.groupby(["store_nbr", "family"])["sales"]
            .transform(
                lambda x: x.shift(1).rolling(28).mean()
            )
        )

        current_features = combined[
            combined["date"] == date
        ].copy()

        current_features = current_features.dropna(
            subset=features
        )

        current_features["predicted_sales"] = model.predict(
            current_features[features]
        )

        future_rows.append(
            current_features[
                [
                    "date",
                    "store_nbr",
                    "family",
                    "onpromotion",
                    "predicted_sales"
                ]
            ]
        )

        predictions_for_history = current_features[
            ["date", "store_nbr", "family", "predicted_sales"]
        ].rename(
            columns={"predicted_sales": "sales"}
        )

        history = pd.concat(
            [
                history,
                predictions_for_history[
                    ["date", "store_nbr", "family", "sales"]
                ].merge(
                    current[
                        ["date", "store_nbr", "family", "onpromotion"]
                    ],
                    on=["date", "store_nbr", "family"],
                    how="left"
                )
            ],
            ignore_index=True
        )

    forecast = pd.concat(
        future_rows,
        ignore_index=True
    )

    forecast.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Future forecast created successfully.")
    print("Rows:", len(forecast))
    print(
        "Date range:",
        forecast["date"].min().date(),
        "to",
        forecast["date"].max().date()
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    create_future_forecast()