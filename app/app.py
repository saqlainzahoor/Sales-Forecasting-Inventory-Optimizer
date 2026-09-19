import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting & Inventory Optimizer",
    page_icon="📦",
    layout="wide"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

FORECAST_FILE = BASE_DIR / "data" / "processed" / "future_forecast.csv"
INVENTORY_FILE = BASE_DIR / "data" / "processed" / "inventory_recommendations.csv"
VALIDATION_FILE = BASE_DIR / "data" / "processed" / "validation_forecast.csv"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "data" / "processed" / "feature_importance.csv"


# Hugging Face model information
HF_REPO_ID = "saqlainzahoorai/sales-forecasting-random-forest"
HF_MODEL_FILENAME = "random_forest_sales_forecaster.joblib"


# --------------------------------------------------
# Load Data
# --------------------------------------------------

@st.cache_data
def load_data():

    forecast = pd.read_csv(FORECAST_FILE)
    inventory = pd.read_csv(INVENTORY_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)

    forecast["date"] = pd.to_datetime(forecast["date"])
    validation["date"] = pd.to_datetime(validation["date"])

    return (
        forecast,
        inventory,
        validation,
        feature_importance
    )


(
    forecast_df,
    inventory_df,
    validation_df,
    feature_importance_df
) = load_data()


# --------------------------------------------------
# Load Model from Hugging Face
# --------------------------------------------------

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME
    )

    return joblib.load(model_path)


model = load_model()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("Sales Forecasting & Inventory Optimizer")

st.markdown(
    """
    **An end-to-end machine learning dashboard for demand forecasting,
    inventory analysis, and data-driven replenishment recommendations.**
    """
)


# --------------------------------------------------
# Sidebar Filters
# --------------------------------------------------

st.sidebar.header("Filters")

store_list = sorted(
    forecast_df["store_nbr"].unique()
)

family_list = sorted(
    forecast_df["family"].unique()
)

selected_store = st.sidebar.selectbox(
    "Store",
    store_list
)

selected_family = st.sidebar.selectbox(
    "Product Family",
    family_list
)


# --------------------------------------------------
# Selected Forecast Data
# --------------------------------------------------

selected_forecast = forecast_df[
    (forecast_df["store_nbr"] == selected_store)
    &
    (forecast_df["family"] == selected_family)
].copy()


selected_inventory = inventory_df[
    (inventory_df["store_nbr"] == selected_store)
    &
    (inventory_df["family"] == selected_family)
].copy()


# --------------------------------------------------
# Forecast Metrics
# --------------------------------------------------

avg_daily_forecast = selected_forecast[
    "predicted_sales"
].mean()

forecast_total = selected_forecast[
    "predicted_sales"
].sum()


current_stock = (
    selected_inventory["current_stock"].iloc[0]
    if not selected_inventory.empty
    else 0
)


recommended_order = (
    selected_inventory["recommended_order_qty"].iloc[0]
    if not selected_inventory.empty
    else 0
)


# --------------------------------------------------
# Forecast Overview
# --------------------------------------------------

st.header("Forecast Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Average Daily Forecast",
        f"{avg_daily_forecast:,.2f}"
    )

with col2:
    st.metric(
        "16-Day Forecast",
        f"{forecast_total:,.0f}"
    )

with col3:
    st.metric(
        "Current Stock",
        f"{current_stock:,.0f}"
    )

with col4:
    st.metric(
        "Recommended Order",
        f"{recommended_order:,.0f}"
    )


# --------------------------------------------------
# Demand Forecast
# --------------------------------------------------

st.header("Demand Forecast")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    selected_forecast["date"],
    selected_forecast["predicted_sales"],
    marker="o"
)

ax.set_title(
    f"16-Day Demand Forecast — "
    f"Store {selected_store} | {selected_family}"
)

ax.set_xlabel("Date")
ax.set_ylabel("Predicted Sales")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)


# --------------------------------------------------
# Inventory Decision
# --------------------------------------------------

st.header("Inventory Decision")

if not selected_inventory.empty:

    inventory_row = selected_inventory.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Reorder Point",
            f"{inventory_row['reorder_point']:,.0f}"
        )

    with col2:
        st.metric(
            "Current Stock",
            f"{inventory_row['current_stock']:,.0f}"
        )

    with col3:
        st.metric(
            "Stock Gap",
            f"{inventory_row['stock_gap']:,.0f}"
        )

    with col4:
        st.metric(
            "Recommended Order",
            f"{inventory_row['recommended_order_qty']:,.0f}"
        )

    status = inventory_row["inventory_status"]

    if status == "REORDER REQUIRED":

        st.warning(
            "Reorder is required based on the current inventory assumptions."
        )

    else:

        st.success(
            "Current inventory level is above the calculated reorder point."
        )


# --------------------------------------------------
# Inventory Methodology & Assumptions
# --------------------------------------------------

st.header("Inventory Methodology & Assumptions")

with st.expander(
    "View Inventory Calculation Methodology",
    expanded=False
):

    st.markdown(
        """
        The inventory optimization layer converts demand forecasts into
        actionable replenishment recommendations.

        Because actual warehouse inventory data is not included in the
        dataset, the current stock level is estimated using predefined
        business assumptions.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Lead Time",
            "3 Days"
        )

    with col2:

        st.metric(
            "Safety Stock",
            "20%"
        )

    with col3:

        st.metric(
            "Current Stock Assumption",
            "2 Days of Demand"
        )

    st.markdown("### Calculation Logic")

    st.markdown(
        """
        **Average Daily Demand**

        Calculated from the forecasted demand for the selected
        store and product family.

        **Lead-Time Demand**

        Average Daily Demand × Lead Time

        **Safety Stock**

        Lead-Time Demand × 20%

        **Reorder Point**

        Lead-Time Demand + Safety Stock

        **Current Stock**

        Average Daily Demand × 2 Days

        **Recommended Order Quantity**

        Maximum of:

        Reorder Point − Current Stock

        or

        0
        """
    )

    st.info(
        "Important: Current Stock is an assumed value because actual "
        "inventory data is not included in the dataset. In a real-world "
        "deployment, this value should be connected to an ERP or "
        "inventory management system."
    )


# --------------------------------------------------
# Business Insight
# --------------------------------------------------

st.header("Business Insight")

if recommended_order > 0:

    st.write(
        f"""
        Based on the forecasted demand, Store {selected_store} for
        {selected_family} has a recommended replenishment quantity of
        approximately {recommended_order:,.0f} units.
        """
    )

else:

    st.write(
        f"""
        Based on the current inventory assumptions, Store {selected_store}
        for {selected_family} does not require immediate replenishment.
        """
    )


# --------------------------------------------------
# Feature Importance
# --------------------------------------------------

st.header("Model Feature Importance")

top_features = feature_importance_df.head(10).copy()

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

ax.set_xlabel("Importance")
ax.set_title(
    "Top 10 Features Used by the Random Forest Model"
)

plt.tight_layout()

st.pyplot(fig)

st.caption(
    "Feature importance represents the model's relative use of each feature "
    "during prediction. It does not imply a causal relationship."
)


# --------------------------------------------------
# Model Validation Performance
# --------------------------------------------------

st.header("Model Validation Performance")

validation_actual = validation_df["sales"]

validation_predicted = validation_df["predicted_sales"]

global_mae = (
    validation_df["error"].abs().mean()
    if "error" in validation_df.columns
    else (
        validation_actual - validation_predicted
    ).abs().mean()
)


global_rmse = (
    (
        (
            validation_actual - validation_predicted
        ) ** 2
    ).mean()
) ** 0.5


col1, col2 = st.columns(2)


# --------------------------------------------------
# Global Performance
# --------------------------------------------------

with col1:

    st.subheader("Global Model Performance")

    st.metric(
        "Validation MAE",
        f"{global_mae:,.2f}"
    )

    st.metric(
        "Validation RMSE",
        f"{global_rmse:,.2f}"
    )


# --------------------------------------------------
# Selected Performance
# --------------------------------------------------

with col2:

    selected_validation = validation_df[
        (validation_df["store_nbr"] == selected_store)
        &
        (validation_df["family"] == selected_family)
    ].copy()

    if not selected_validation.empty:

        selected_actual = selected_validation["sales"]

        selected_predicted = selected_validation[
            "predicted_sales"
        ]

        selected_mae = (
            selected_actual - selected_predicted
        ).abs().mean()

        selected_rmse = (
            (
                (
                    selected_actual - selected_predicted
                ) ** 2
            ).mean()
        ) ** 0.5

        st.subheader(
            "Selected Store / Product Family"
        )

        st.metric(
            "Selected MAE",
            f"{selected_mae:,.2f}"
        )

        st.metric(
            "Selected RMSE",
            f"{selected_rmse:,.2f}"
        )


# --------------------------------------------------
# Actual vs Predicted
# --------------------------------------------------

st.header("Actual Sales vs Predicted Sales")

if not selected_validation.empty:

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        selected_validation["date"],
        selected_validation["sales"],
        label="Actual Sales"
    )

    ax.plot(
        selected_validation["date"],
        selected_validation["predicted_sales"],
        label="Predicted Sales"
    )

    ax.set_title(
        f"Actual vs Predicted Sales — "
        f"Store {selected_store} | {selected_family}"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


# --------------------------------------------------
# Forecast Error Analysis
# --------------------------------------------------

st.header("Forecast Error Analysis")

if not selected_validation.empty:

    selected_validation["error"] = (
        selected_validation["sales"]
        - selected_validation["predicted_sales"]
    )

    selected_validation["absolute_error"] = (
        selected_validation["error"].abs()
    )

    mean_error = (
        selected_validation["error"].mean()
    )

    mae = (
        selected_validation["absolute_error"].mean()
    )

    median_absolute_error = (
        selected_validation["absolute_error"].median()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Mean Error",
            f"{mean_error:,.2f}"
        )

    with col2:

        st.metric(
            "Mean Absolute Error",
            f"{mae:,.2f}"
        )

    with col3:

        st.metric(
            "Median Absolute Error",
            f"{median_absolute_error:,.2f}"
        )


    # --------------------------------------------------
    # Error Over Time
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        selected_validation["date"],
        selected_validation["error"]
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_title(
        f"Prediction Error Over Time — "
        f"Store {selected_store} | {selected_family}"
    )

    ax.set_xlabel("Date")

    ax.set_ylabel(
        "Prediction Error "
        "(Actual − Predicted)"
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)


    # --------------------------------------------------
    # Error Interpretation
    # --------------------------------------------------

    if mean_error > 0:

        st.info(
            "The model slightly underestimated demand for the selected "
            "store and product family during the validation period."
        )

    elif mean_error < 0:

        st.info(
            "The model slightly overestimated demand for the selected "
            "store and product family during the validation period."
        )

    else:

        st.info(
            "The model showed minimal overall bias for the selected "
            "store and product family during the validation period."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Sales Forecasting & Inventory Optimizer | "
    "Machine Learning Portfolio Project"
)

st.markdown("---")

st.caption("Developed by Saqlain Zahoor")
