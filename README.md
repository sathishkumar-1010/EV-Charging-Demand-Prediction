# EV Charging Demand Prediction and Usage Pattern Analysis

## Project Overview

This project uses Machine Learning and Time-Series Forecasting to analyze electric vehicle (EV) charging patterns and predict charging demand.

The project analyzes historical EV charging sessions to identify usage patterns, predict demand, classify charging sessions into High Demand and Normal Demand, and forecast future charging demand.

An interactive Streamlit dashboard is developed to visualize the analysis and demonstrate the prediction models.

## Project Objectives

- Analyze EV charging usage patterns across different locations, charger types, and companies.
- Perform Exploratory Data Analysis (EDA) to identify demand trends.
- Engineer time-based and historical usage features.
- Predict EV charging demand using regression algorithms.
- Classify charging demand into High Demand and Normal Demand.
- Forecast hourly and daily charging demand.
- Explain model predictions using SHAP and LIME.
- Build an interactive dashboard using Streamlit.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| ML Algorithms | Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, and others |
| Time-Series Forecasting | Moving Average, Exponential Smoothing |
| Explainable AI | SHAP, LIME |
| Dashboard | Streamlit |
| Model Storage | Pickle |
| Development Environment | Visual Studio Code |

## Project Features

### 1. Data Cleaning and Preprocessing
- Removed duplicate records.
- Validated charging session timestamps.
- Calculated charging duration using timestamps.
- Prepared cleaned datasets for analysis and modeling.

### 2. Exploratory Data Analysis (EDA)
- Demand distribution analysis.
- Location-wise charging demand.
- Charger type and company analysis.
- Hourly, daily, and monthly usage patterns.
- Duration and demand relationship analysis.
- Correlation and outlier analysis.

### 3. Feature Engineering
Created time-based features such as:
- Hour
- Day
- Month
- Day of Week
- Weekend indicator
- Time of Day

Also generated historical usage features based on previous charging sessions.

### 4. Regression – Demand Prediction

Regression algorithms evaluated:
- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

**Best Model: LightGBM Regressor**

Final test results:

| Evaluation Metric | Score |
|---|---:|
| MAE | 10.1488 |
| MSE | 172.7856 |
| RMSE | 13.1448 |
| R² Score | 0.1058 |

### 5. Classification – Demand Prediction

Classifies charging demand into:
- High Demand
- Normal Demand

Algorithms evaluated include Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, and LightGBM.

**Best Model: LightGBM Classifier**

Final test results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 64.10% |
| Precision | 62.09% |
| Recall | 67.72% |
| F1 Score | 64.78% |
| ROC-AUC | 68.74% |

### 6. Time-Series Forecasting

Implemented:
- Hourly demand aggregation.
- Daily demand aggregation.
- Stationarity testing using the Augmented Dickey-Fuller (ADF) test.
- Moving Average forecasting.
- Exponential Smoothing forecasting.

Forecast evaluation:

| Forecasting Model | MAE | RMSE |
|---|---:|---:|
| Moving Average | 129.46 | 152.59 |
| Exponential Smoothing | 65.48 | 85.72 |

Exponential Smoothing achieved lower MAE and RMSE than Moving Average in the evaluated test period.

### 7. Model Explainability

Used Explainable AI techniques to understand model predictions:

- Feature Importance
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)

These techniques help identify the features influencing charging demand predictions.

## Project Structure

```text
EV_Charging_Demand_Prediction/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── best_regression_model.pkl
│   ├── best_classification_model.pkl
│   ├── demand_threshold.pkl
│   └── preprocessor.pkl
│
├── reports/
│   ├── eda/
│   ├── explainability/
│   ├── time_series/
│   ├── regression_model_results.csv
│   └── classification_test_results.csv
│
├── src/
│   ├── inspect_dataset.py
│   ├── clean_dataset.py
│   ├── eda_analysis.py
│   ├── feature_engineering.py
│   ├── prepare_data.py
│   ├── preprocess_data.py
│   ├── prepare_targets.py
│   ├── train_regression.py
│   ├── train_classification.py
│   ├── time_series_forecasting.py
│   └── model_explainability.py
│
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sathishkumar-1010/EV-Charging-Demand-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd EV-Charging-Demand-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit xgboost lightgbm shap lime statsmodels joblib
```

### 5. Dataset setup

The original raw dataset is not included in this GitHub repository.

Place the dataset file named `ChargingRecords.csv` inside:

```text
data/raw/ChargingRecords.csv
```

### 6. Run the Streamlit dashboard

From the project root:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## Model Performance Note

The regression and classification models demonstrate the feasibility of predicting EV charging demand using historical charging patterns. Their current test performance is moderate, and further improvements in feature engineering and model optimization may improve prediction accuracy.

## Future Enhancements

- Improve regression and classification performance.
- Explore additional time-series forecasting techniques.
- Add real-time EV charging data integration.
- Improve dashboard interactivity.
- Deploy the dashboard to a cloud platform.

## Author

**Sathish Kumar R.**

B.Tech – Artificial Intelligence and Data Science (2025)

GitHub: [sathishkumar-1010](https://github.com/sathishkumar-1010)

## Project Repository

[EV Charging Demand Prediction – GitHub](https://github.com/sathishkumar-1010/EV-Charging-Demand-Prediction)
