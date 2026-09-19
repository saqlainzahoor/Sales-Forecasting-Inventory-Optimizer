import pandas as pd


def create_features(input_path, output_path):
    # -----------------------------------
    # Load Data
    # -----------------------------------
    df = pd.read_csv(
        input_path,
        usecols=[
            "date",
            "store_nbr",
            "family",
            "sales",
            "onpromotion"
        ]
    )

    df["date"] = pd.to_datetime(df["date"])

    # -----------------------------------
    # Sort Data
    # -----------------------------------
    df = df.sort_values(
        ["store_nbr", "family", "date"]
    ).reset_index(drop=True)

    # -----------------------------------
    # Time Features
    # -----------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # -----------------------------------
    # Lag Features
    # -----------------------------------
    group = df.groupby(
        ["store_nbr", "family"],
        sort=False
    )["sales"]

    df["lag_1"] = group.shift(1)
    df["lag_7"] = group.shift(7)
    df["lag_14"] = group.shift(14)
    df["lag_28"] = group.shift(28)

    # -----------------------------------
    # Rolling Features
    # IMPORTANT:
    # shift(1) prevents data leakage
    # -----------------------------------
    df["rolling_7"] = (
        df.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["rolling_14"] = (
        df.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .transform(
            lambda x: x.shift(1).rolling(14).mean()
        )
    )

    df["rolling_28"] = (
        df.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .transform(
            lambda x: x.shift(1).rolling(28).mean()
        )
    )

    # -----------------------------------
    # Save Feature Dataset
    # -----------------------------------
    df.to_csv(
        output_path,
        index=False
    )

    # -----------------------------------
    # Information
    # -----------------------------------
    print("Feature dataset created successfully.")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":

    create_features(
        "data/raw/train.csv",
        "data/processed/train_features.csv"
    )