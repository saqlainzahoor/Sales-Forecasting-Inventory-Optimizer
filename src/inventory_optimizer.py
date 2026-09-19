import pandas as pd


FORECAST_PATH = "data/processed/future_forecast.csv"
OUTPUT_PATH = "data/processed/inventory_recommendations.csv"

LEAD_TIME_DAYS = 3
SAFETY_STOCK_RATE = 0.20
CURRENT_STOCK_DAYS = 2


def create_inventory_recommendations():

    df = pd.read_csv(FORECAST_PATH)
    df["date"] = pd.to_datetime(df["date"])

    daily_demand = (
        df.groupby(["store_nbr", "family"])["predicted_sales"]
        .mean()
        .reset_index(name="avg_daily_demand")
    )

    daily_demand["lead_time_demand"] = (
        daily_demand["avg_daily_demand"] * LEAD_TIME_DAYS
    )

    daily_demand["safety_stock"] = (
        daily_demand["lead_time_demand"] * SAFETY_STOCK_RATE
    )

    daily_demand["reorder_point"] = (
        daily_demand["lead_time_demand"]
        + daily_demand["safety_stock"]
    )

    daily_demand["current_stock"] = (
        daily_demand["avg_daily_demand"] * CURRENT_STOCK_DAYS
    )

    daily_demand["stock_gap"] = (
        daily_demand["reorder_point"]
        - daily_demand["current_stock"]
    ).clip(lower=0)

    daily_demand["reorder_required"] = (
        daily_demand["current_stock"]
        < daily_demand["reorder_point"]
    )

    daily_demand["recommended_order_qty"] = (
        daily_demand["stock_gap"]
    ).round(0)

    daily_demand["inventory_status"] = (
        daily_demand["reorder_required"]
        .map({
            True: "REORDER REQUIRED",
            False: "STOCK LEVEL OK"
        })
    )

    daily_demand["lead_time_days"] = LEAD_TIME_DAYS
    daily_demand["safety_stock_rate"] = SAFETY_STOCK_RATE
    daily_demand["current_stock_days"] = CURRENT_STOCK_DAYS

    daily_demand.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Inventory recommendations created successfully.")
    print("Rows:", len(daily_demand))
    print("Saved:", OUTPUT_PATH)

    print("\nInventory status:")
    print(
        daily_demand["inventory_status"]
        .value_counts()
        .to_string()
    )

    print("\nTop 10 reorder recommendations:")
    print(
        daily_demand
        .sort_values("recommended_order_qty", ascending=False)
        .head(10)
        [
            [
                "store_nbr",
                "family",
                "avg_daily_demand",
                "current_stock",
                "reorder_point",
                "stock_gap",
                "recommended_order_qty",
                "inventory_status"
            ]
        ]
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    create_inventory_recommendations()