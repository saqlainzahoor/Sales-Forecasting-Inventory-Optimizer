# Sales Forecasting & Inventory Optimizer

**Author:** Saqlain Zahoor

An end-to-end machine learning project for **demand forecasting, inventory analysis, and data-driven replenishment recommendations**.

The project uses historical retail sales data to forecast future product demand and translate those forecasts into practical inventory decisions.

---

## 📌 Project Overview

Retail businesses need accurate demand forecasts to maintain appropriate inventory levels.

Overstocking can increase holding costs, while understocking can lead to missed sales and poor customer experience.

This project addresses this problem through an end-to-end machine learning pipeline:

**Historical Sales → Data Preparation → Feature Engineering → Time-Series Validation → Demand Forecasting → Inventory Optimization → Replenishment Recommendation → Business Dashboard**

The project goes beyond model prediction by connecting demand forecasts with a practical inventory decision layer.

---

## 🎯 Business Problem

A retail business needs to answer questions such as:

* How much demand should we expect in the coming days?
* Which products may require replenishment?
* How much inventory should be ordered?
* How accurately is the forecasting model performing?
* Which features contribute most to model predictions?
* Where are forecasting errors concentrated?

The goal of this project is to build a machine learning system that helps answer these questions using historical sales data.

---

## 💡 Solution

The system combines:

1. Historical sales analysis
2. Data preparation
3. Time-based feature engineering
4. Lag and rolling demand features
5. Chronological time-series validation
6. Machine learning demand forecasting
7. Forecast error analysis
8. Model explainability
9. Inventory optimization
10. Reorder point calculation
11. Recommended order quantity
12. Interactive Streamlit dashboard

This creates a complete workflow from **prediction to business action**.

---

## 📊 Dataset

The project uses the **Store Sales - Time Series Forecasting** dataset from Kaggle.

### Dataset Characteristics

* **3,000,888** training records
* **54** stores
* **33** product families
* Daily sales data
* Promotion information
* Store metadata
* Oil price information
* Holiday information
* Transaction information

### Training Period

**2013-01-01 → 2017-08-15**

### Forecast Horizon

**2017-08-16 → 2017-08-31**

The forecast horizon contains:

* **16 days**
* **54 stores**
* **33 product families**
* **28,512 store-product observations**

---

## 🔎 Exploratory Data Analysis

The exploratory analysis examined:

* Sales distribution
* Zero-sales frequency
* Sales trends over time
* Weekly demand patterns
* Monthly demand patterns
* Store-level demand
* Product-family demand
* Promotion patterns
* Holiday information
* Oil prices
* Transaction activity

### Important Observations

The dataset contains a large number of zero-sales observations and a strongly right-skewed sales distribution.

Recent historical demand showed strong predictive value, particularly through weekly lag and rolling-demand features.

The analysis also showed meaningful differences in demand across stores and product families.

---

## 🛠️ Feature Engineering

The forecasting model uses time-based, store-level, promotion, lag, and rolling-demand features.

### Date Features

* `year`
* `month`
* `day`
* `day_of_week`
* `week_of_year`
* `quarter`
* `is_weekend`

### Lag Features

* `lag_1`
* `lag_7`
* `lag_14`
* `lag_28`

### Rolling Features

* `rolling_7`
* `rolling_14`
* `rolling_28`

### Additional Features

* `store_nbr`
* `onpromotion`

Rolling features were created using shifted historical demand to reduce the risk of target leakage.

---

## ⏱️ Time-Series Validation

Because this is a forecasting problem, the project does **not** use a random train-test split.

Instead, the data is divided chronologically.

### Training Period

**Before 2017-07-01**

### Validation Period

**2017-07-01 → 2017-08-15**

This approach better represents a real forecasting environment because future observations are not available during model training.

---

## 🤖 Models & Baselines

The project evaluates simple historical-demand baselines alongside machine learning models.

### Baseline 1 — 7-Day Lag

* **MAE:** 88.63
* **RMSE:** 331.07

### Baseline 2 — 28-Day Rolling Mean

* **MAE:** 105.71
* **RMSE:** 372.01

### Machine Learning Models

The project evaluated:

* Random Forest
* HistGradientBoosting

Random Forest was selected for the final forecasting pipeline based on the validation experiments conducted during development.

### Final Random Forest Configuration

* **Number of trees:** 75
* **Maximum depth:** 18
* **Minimum samples per leaf:** 3
* **Training sample:** 500,000 rows
* **Parallel jobs:** 2

The training sample was used as a practical memory-management strategy because the complete dataset is large.

---

## 📈 Model Validation Results

The final Random Forest model achieved the following results on the chronological validation period:

| Metric |     Result |
| ------ | ---------: |
| MAE    |  **59.82** |
| RMSE   | **218.57** |

The model was evaluated against historical-demand baselines rather than being assessed only in isolation.

### Validation Interpretation

MAE measures the average absolute forecasting error, while RMSE gives greater weight to larger errors.

Using both metrics provides a broader view of forecasting performance.

---

## 🧠 Model Explainability

Feature importance was used to understand which variables the Random Forest relied on most during prediction.

### Top Features

| Feature     | Importance |
| ----------- | ---------: |
| `rolling_7` |     0.7880 |
| `lag_7`     |     0.0846 |
| `lag_14`    |     0.0682 |
| `lag_1`     |     0.0182 |
| `lag_28`    |     0.0117 |

The model relies heavily on recent demand history, especially the 7-day rolling demand feature.

> Feature importance indicates how the model uses features for prediction. It does not establish a causal relationship.

---

## 🔮 Future Demand Forecast

The final model generates recursive forecasts for the 16-day forecast horizon.

### Forecast Period

**2017-08-16 → 2017-08-31**

### Forecast Coverage

* 54 stores
* 33 product families
* 28,512 forecast observations

The forecast is generated recursively, meaning previous predictions are used to construct historical-demand features for subsequent forecast days.

The resulting demand forecast is then passed to the inventory optimization layer.

---

## 📦 Inventory Optimization

The inventory layer converts demand forecasts into replenishment recommendations.

Because actual warehouse inventory data is not included in the dataset, the project uses explicit business assumptions.

### Business Assumptions

| Parameter     |                        Assumption |
| ------------- | --------------------------------: |
| Lead Time     |                            3 days |
| Safety Stock  |           20% of lead-time demand |
| Current Stock | 2 days of average forecast demand |

### Calculation Logic

**Average Daily Demand**

Average predicted demand across the forecast horizon.

**Lead-Time Demand**

`Average Daily Demand × Lead Time`

**Safety Stock**

`Lead-Time Demand × Safety Stock Rate`

**Reorder Point**

`Lead-Time Demand + Safety Stock`

**Current Stock**

`Average Daily Demand × Current Stock Days`

**Recommended Order Quantity**

`max(Reorder Point − Current Stock, 0)`

### Important Limitation

The current stock value is an **assumption**, not actual company inventory data.

For real-world deployment, this value should be connected to an ERP system, warehouse management system, or inventory database.

The same applies to lead time and safety-stock assumptions, which should ideally be based on real operational data.

---

## 💼 Business Insights

The system can provide:

* Expected future demand
* Reorder point
* Estimated current stock
* Stock gap
* Recommended order quantity
* Forecasting error
* Store-level model performance
* Product-level model performance
* Model feature importance

This creates a connection between machine learning predictions and operational inventory decisions.

---

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard designed to connect forecasting results with inventory decisions.

### Forecast Overview

Displays:

* Average Daily Forecast
* 16-Day Forecast
* Current Stock
* Recommended Order

### Demand Forecast

Interactive forecast visualization based on the selected:

* Store
* Product Family

### Inventory Decision

Displays:

* Reorder Point
* Current Stock
* Stock Gap
* Recommended Order

### Inventory Methodology

The dashboard explains:

* Lead Time
* Safety Stock
* Current Stock Assumption
* Reorder Point
* Recommended Order Calculation

### Model Explainability

Displays the top model features and their relative importance.

### Model Validation

Displays:

* Global MAE
* Global RMSE
* Selected Store/Product Family MAE
* Selected Store/Product Family RMSE

### Error Analysis

Displays:

* Mean Error
* Mean Absolute Error
* Median Absolute Error
* Prediction Error Over Time

---

## 🖥️ Project Architecture

```text
Historical Sales Data
        │
        ▼
Data Preparation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Time-Series Validation
        │
        ▼
Demand Forecasting Model
        │
        ▼
Future Demand Forecast
        │
        ▼
Inventory Optimization
        │
        ▼
Reorder Recommendation
        │
        ▼
Streamlit Dashboard
```

---

## 📁 Project Structure

```text
Sales-Forecasting-Inventory-Optimizer/
│
├── app/
│   └── app.py
│
├── data/
│   └── processed/
│       ├── future_forecast.csv
│       ├── inventory_recommendations.csv
│       ├── feature_importance.csv
│       ├── error_analysis.csv
│       └── validation_forecast.csv
│
├── notebooks/
│
├── src/
│   ├── feature_engineering.py
│   ├── forecasting.py
│   ├── future_forecast.py
│   └── inventory_optimizer.py
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Model Storage

The trained Random Forest model is **not stored directly in the GitHub repository** because of GitHub's file-size limitation.

Instead, the model is hosted in a dedicated Hugging Face model repository and downloaded by the Streamlit application at runtime.

**Hugging Face Model Repository:**

`saqlainzahoorai/sales-forecasting-random-forest`

This approach keeps the GitHub repository lightweight while allowing the deployed application to load the trained model when required.

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Hugging Face Hub
* Kaggle Dataset
* Git
* GitHub

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/saqlainzahoor/Sales-Forecasting-Inventory-Optimizer.git
```

### 2. Move Into the Project Directory

```bash
cd Sales-Forecasting-Inventory-Optimizer
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment on Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app/app.py
```

The dashboard will open in your browser.

---

## 🌐 Live Demo

The project is deployed using Streamlit Community Cloud.

**Live Application:**

https://sales-forecasting-inventory-optimizer.streamlit.app/

---

## 🧪 Model Evaluation

The project evaluates forecasting performance using:

### MAE — Mean Absolute Error

MAE measures the average absolute difference between actual and predicted demand.

Lower MAE indicates smaller average forecasting errors.

### RMSE — Root Mean Squared Error

RMSE gives greater weight to larger prediction errors.

Lower RMSE indicates fewer large forecasting errors.

Both metrics are used to provide a broader view of forecasting performance.

---

## ⚠️ Limitations

This project has several important limitations:

1. Actual inventory levels are not available in the dataset.
2. Current stock is therefore based on a business assumption.
3. Lead time is assumed to be 3 days.
4. Safety stock is assumed to be 20% of lead-time demand.
5. Recursive forecasting can accumulate prediction errors across the forecast horizon.
6. Feature importance should not be interpreted as causal evidence.
7. Transaction data contains missing observations and should not automatically be treated as zero activity.
8. The model is trained on historical patterns and may not fully capture unexpected future events.
9. Inventory recommendations are illustrative because real-time stock, supplier lead times, and service-level requirements are not available.

---

## 🔮 Future Improvements

Potential improvements include:

* XGBoost / LightGBM comparison
* Advanced time-series models
* More extensive hyperparameter optimization
* Walk-forward validation
* Prediction intervals
* Probabilistic forecasting
* Dynamic safety-stock calculation
* Service-level-based inventory optimization
* Actual ERP inventory integration
* Supplier lead-time integration
* Automated reorder alerts
* Multi-location inventory optimization
* API deployment
* Cloud-based production architecture
* Real-time forecasting pipeline

---

## 🎯 Portfolio Skills Demonstrated

This project demonstrates practical experience in:

* Data Cleaning
* Exploratory Data Analysis
* Time-Series Forecasting
* Feature Engineering
* Machine Learning
* Model Comparison
* Model Evaluation
* Error Analysis
* Model Explainability
* Inventory Optimization
* Business Decision Support
* Streamlit Dashboard Development
* Hugging Face Model Deployment
* Git & GitHub
* Cloud Deployment
* End-to-End ML Project Development

---

## 👨‍💻 Project Purpose

This project was developed as a portfolio project to demonstrate how machine learning can be transformed from a predictive model into a practical business decision-support system.

The focus is not only on forecasting accuracy, but also on connecting predictions with inventory planning and operational decision-making.

The project demonstrates an end-to-end workflow covering:

**Data → Machine Learning → Forecasting → Business Logic → Inventory Recommendation → Interactive Dashboard → Deployment**

---
