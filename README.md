# Sales Forecasting & Inventory Optimizer

An end-to-end machine learning project for **demand forecasting, inventory analysis, and data-driven replenishment recommendations**.

The project uses historical retail sales data to forecast future product demand and translate those forecasts into practical inventory decisions.

---

## 📌 Project Overview

Retail businesses need accurate demand forecasts to maintain the right inventory levels.

Overstocking can increase holding costs, while understocking can lead to missed sales and poor customer experience.

This project addresses that problem through an end-to-end machine learning pipeline:

**Historical Sales → Data Preparation → Feature Engineering → Time-Series Validation → Demand Forecasting → Inventory Optimization → Replenishment Recommendation → Business Dashboard**

The project goes beyond model prediction by connecting the forecast to a practical inventory decision layer.

---

## 🎯 Business Problem

A retail business needs to answer questions such as:

* How much demand should we expect in the coming days?
* Which products may require replenishment?
* How much inventory should be ordered?
* How accurately is the forecasting model performing?
* Which factors contribute most to the model's predictions?
* Where are forecasting errors concentrated?

The goal of this project is to build a machine learning system that helps answer these questions using historical sales data.

---

## 💡 Solution

The system combines:

1. Historical sales analysis
2. Time-based feature engineering
3. Lag and rolling demand features
4. Time-series validation
5. Machine learning demand forecasting
6. Forecast error analysis
7. Inventory optimization
8. Reorder point calculation
9. Recommended order quantity
10. Interactive Streamlit dashboard

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

The forecasting horizon contains **16 days** and **28,512 store-product observations**.

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

Recent historical demand also showed strong predictive value, particularly through weekly lag and rolling-demand features.

---

## 🛠️ Feature Engineering

The forecasting model uses time-based and historical demand features.

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

Instead, historical data is divided chronologically.

### Training Period

Before:

**2017-07-01**

### Validation Period

**2017-07-01 → 2017-08-15**

This approach better represents how the model would operate in a real forecasting environment, where future observations are not available during training.

---

## 🤖 Models & Baselines

The project evaluates simple forecasting baselines alongside machine learning models.

### Baseline Models

#### 7-Day Lag Baseline

* MAE: **88.63**
* RMSE: **331.07**

#### 28-Day Rolling Baseline

* MAE: **105.71**
* RMSE: **372.01**

### Machine Learning Models

The project evaluated:

* Random Forest
* HistGradientBoosting

Random Forest was selected for the final forecasting pipeline based on the validation experiments conducted during development.

### Final Random Forest Configuration

* Number of trees: **75**
* Maximum depth: **18**
* Minimum samples per leaf: **3**
* Training sample: **500,000 rows**
* Parallel jobs: **2**

---

## 📈 Model Validation Results

The final Random Forest model achieved the following validation results:

| Metric |     Result |
| ------ | ---------: |
| MAE    |  **59.82** |
| RMSE   | **218.57** |

Validation performance was measured on the chronological validation period.

The model was also compared against historical-demand baselines rather than being evaluated only in isolation.

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

The forecast is then passed to the inventory optimization layer.

---

## 📦 Inventory Optimization

The inventory layer converts demand forecasts into replenishment recommendations.

### Business Assumptions

Because actual warehouse inventory data is not included in the dataset, the project uses explicit assumptions:

| Parameter     |                        Assumption |
| ------------- | --------------------------------: |
| Lead Time     |                            3 days |
| Safety Stock  |           20% of lead-time demand |
| Current Stock | 2 days of average forecast demand |

### Calculation Logic

**Lead-Time Demand**

Average Daily Demand × Lead Time

**Safety Stock**

Lead-Time Demand × Safety Stock Rate

**Reorder Point**

Lead-Time Demand + Safety Stock

**Recommended Order Quantity**

Maximum of:

`Reorder Point − Current Stock`

or

`0`

### Important Limitation

The current stock value is an **assumption**, not actual company inventory data.

For real-world deployment, this value should be connected to an ERP, warehouse management system, or inventory database.

---

## 💼 Business Insights

The system can provide:

* Expected future demand
* Reorder point
* Current estimated stock
* Stock gap
* Recommended order quantity
* Forecasting error
* Store/product-level model performance

This creates a connection between machine learning predictions and operational inventory decisions.

---

## 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard with:

### Forecast Overview

* Average Daily Forecast
* 16-Day Forecast
* Current Stock
* Recommended Order

### Demand Forecast

Interactive forecast visualization for the selected:

* Store
* Product Family

### Inventory Decision

Displays:

* Reorder Point
* Current Stock
* Stock Gap
* Recommended Order

### Inventory Methodology

Explains:

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
│   ├── raw/
│   └── processed/
│
├── models/
│   └── random_forest_sales_forecaster.joblib
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
* Kaggle Dataset
* Git
* GitHub

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/saqlainzahoor/Sales-Forecasting-Inventory-Optimizer.git
```

Move into the project directory:

```bash
cd Sales-Forecasting-Inventory-Optimizer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

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

## 🧪 Model Evaluation

The project evaluates forecasting performance using:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted demand.

### RMSE

Root Mean Squared Error gives greater weight to larger prediction errors.

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

---

## 🔮 Future Improvements

Potential improvements include:

* XGBoost / LightGBM comparison
* Advanced time-series models
* Hyperparameter optimization
* Walk-forward validation
* Prediction intervals
* Probabilistic forecasting
* Dynamic safety stock
* Service-level-based inventory optimization
* Actual ERP inventory integration
* Supplier lead-time integration
* Automated reorder alerts
* Multi-location inventory optimization
* API deployment
* Cloud-based production deployment

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
* Git & GitHub
* End-to-End ML Project Development

---

## 👨‍💻 Project Purpose

This project was developed as a portfolio project to demonstrate how machine learning can be transformed from a predictive model into a practical business decision-support system.

The focus is not only on forecasting accuracy, but also on connecting predictions with inventory planning and operational decision-making.
