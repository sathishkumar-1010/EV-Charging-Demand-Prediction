
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


# --------------------------------------------------
# 1. CREATE OUTPUT FOLDERS
# --------------------------------------------------

os.makedirs("data/processed", exist_ok=True)
os.makedirs("reports/time_series", exist_ok=True)
os.makedirs("reports/time_series/charts", exist_ok=True)

print("Starting Time Series Forecasting...")


# --------------------------------------------------
# 2. LOAD PROCESSED DATASET
# --------------------------------------------------

print("\nLoading processed dataset...")

df = pd.read_csv(
    "data/processed/charging_ml_dataset.csv"
)

df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"]
)

df = df.sort_values("StartDatetime")

print("Dataset shape:", df.shape)
print("Date range:")
print(df["StartDatetime"].min())
print(df["StartDatetime"].max())


# --------------------------------------------------
# 3. HOURLY DEMAND AGGREGATION
# --------------------------------------------------

print("\nAggregating demand by hour...")

hourly_data = (
    df.set_index("StartDatetime")["Demand"]
    .resample("h")
    .sum()
)

# Fill hours without charging sessions with zero total demand
hourly_data = hourly_data.asfreq("h", fill_value=0)

hourly_df = hourly_data.reset_index()

hourly_df.columns = [
    "Datetime",
    "TotalDemand"
]

hourly_df.to_csv(
    "data/processed/demand_hourly.csv",
    index=False
)

print("Hourly dataset shape:", hourly_df.shape)
print(hourly_df.head())


# --------------------------------------------------
# 4. DAILY DEMAND AGGREGATION
# --------------------------------------------------

print("\nAggregating demand by day...")

daily_data = (
    df.set_index("StartDatetime")["Demand"]
    .resample("D")
    .sum()
)

daily_df = daily_data.reset_index()

daily_df.columns = [
    "Date",
    "TotalDemand"
]

daily_df.to_csv(
    "data/processed/demand_daily.csv",
    index=False
)

print("Daily dataset shape:", daily_df.shape)
print(daily_df.head())


# --------------------------------------------------
# 5. VISUALIZE HOURLY DEMAND TREND
# --------------------------------------------------

print("\nCreating hourly demand trend chart...")

plt.figure(figsize=(14, 5))

plt.plot(
    hourly_data.index,
    hourly_data.values
)

plt.title("Hourly EV Charging Demand")
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.tight_layout()

plt.savefig(
    "reports/time_series/charts/hourly_demand_trend.png"
)

plt.close()


# --------------------------------------------------
# 6. VISUALIZE DAILY DEMAND TREND
# --------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    daily_data.index,
    daily_data.values
)

plt.title("Daily EV Charging Demand")
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.tight_layout()

plt.savefig(
    "reports/time_series/charts/daily_demand_trend.png"
)

plt.close()


# --------------------------------------------------
# 7. ANALYZE HOURLY SEASONALITY
# --------------------------------------------------

print("\nAnalyzing hourly seasonality...")

hourly_pattern = (
    df.assign(Hour=df["StartDatetime"].dt.hour)
    .groupby("Hour")["Demand"]
    .mean()
)

plt.figure(figsize=(10, 5))

plt.plot(
    hourly_pattern.index,
    hourly_pattern.values,
    marker="o"
)

plt.title("Average EV Charging Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Demand")
plt.xticks(range(24))
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "reports/time_series/charts/hourly_seasonality.png"
)

plt.close()

print(hourly_pattern)


# --------------------------------------------------
# 8. STATIONARITY TEST
# --------------------------------------------------

print("\nPerforming Augmented Dickey-Fuller Test...")

adf_result = adfuller(hourly_data)

adf_statistic = adf_result[0]
p_value = adf_result[1]

print("ADF Statistic:", round(adf_statistic, 4))
print("P-value:", round(p_value, 6))

if p_value < 0.05:
    print("Result: The hourly series appears stationary.")
else:
    print("Result: The hourly series appears non-stationary.")

with open(
    "reports/time_series/stationarity_result.txt",
    "w"
) as file:

    file.write(
        f"ADF Statistic: {adf_statistic}\n"
    )

    file.write(
        f"P-value: {p_value}\n"
    )

    if p_value < 0.05:
        file.write("The hourly series appears stationary.")
    else:
        file.write("The hourly series appears non-stationary.")


# --------------------------------------------------
# 9. TRAIN-TEST SPLIT
# --------------------------------------------------

print("\nSplitting hourly data chronologically...")

split_index = int(len(hourly_data) * 0.80)

train_data = hourly_data.iloc[:split_index]
test_data = hourly_data.iloc[split_index:]

print("Training observations:", len(train_data))
print("Testing observations:", len(test_data))


# --------------------------------------------------
# 10. MOVING AVERAGE FORECASTING
# --------------------------------------------------

print("\nRunning Moving Average Forecasting...")

window_size = 24

history = list(train_data.values)
moving_average_predictions = []

for i in range(len(test_data)):

    prediction = np.mean(history[-window_size:])

    moving_average_predictions.append(prediction)

    # Update history using the forecast, not actual test demand
    history.append(prediction)


# --------------------------------------------------
# 11. EXPONENTIAL SMOOTHING FORECASTING
# --------------------------------------------------

print("\nRunning Exponential Smoothing Forecasting...")

try:

    exponential_model = ExponentialSmoothing(
        train_data,
        trend="add",
        seasonal="add",
        seasonal_periods=24,
        initialization_method="estimated"
    )

    exponential_fit = exponential_model.fit(
        optimized=True
    )

    exponential_predictions = exponential_fit.forecast(
        len(test_data)
    )

    exponential_predictions = np.maximum(
        exponential_predictions.values,
        0
    )

except Exception as error:

    print("Seasonal Exponential Smoothing failed:", error)

    print("Trying Exponential Smoothing without seasonality...")

    exponential_model = ExponentialSmoothing(
        train_data,
        trend="add",
        initialization_method="estimated"
    )

    exponential_fit = exponential_model.fit(
        optimized=True
    )

    exponential_predictions = exponential_fit.forecast(
        len(test_data)
    ).values

    exponential_predictions = np.maximum(
        exponential_predictions,
        0
    )


# --------------------------------------------------
# 12. EVALUATE FORECASTING MODELS
# --------------------------------------------------

print("\nEvaluating forecasting models...")

actual_values = test_data.values

moving_average_mae = mean_absolute_error(
    actual_values,
    moving_average_predictions
)

moving_average_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        moving_average_predictions
    )
)

exponential_mae = mean_absolute_error(
    actual_values,
    exponential_predictions
)

exponential_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        exponential_predictions
    )
)


# --------------------------------------------------
# 13. DISPLAY MODEL COMPARISON
# --------------------------------------------------

forecast_results = pd.DataFrame({

    "Model": [
        "Moving Average",
        "Exponential Smoothing"
    ],

    "MAE": [
        moving_average_mae,
        exponential_mae
    ],

    "RMSE": [
        moving_average_rmse,
        exponential_rmse
    ]

})

print("\n======================================")
print("TIME SERIES FORECASTING RESULTS")
print("======================================")

print(
    forecast_results.to_string(index=False)
)

forecast_results.to_csv(
    "reports/time_series/forecast_results.csv",
    index=False
)


# --------------------------------------------------
# 14. SAVE FORECAST VALUES
# --------------------------------------------------

forecast_df = pd.DataFrame({

    "Datetime": test_data.index,

    "Actual_Demand": actual_values,

    "Moving_Average_Forecast": moving_average_predictions,

    "Exponential_Smoothing_Forecast": exponential_predictions

})

forecast_df.to_csv(
    "reports/time_series/forecast_predictions.csv",
    index=False
)


# --------------------------------------------------
# 15. VISUALIZE FORECAST COMPARISON
# --------------------------------------------------

print("\nCreating forecast comparison chart...")

plt.figure(figsize=(14, 6))

plt.plot(
    test_data.index,
    actual_values,
    label="Actual Demand"
)

plt.plot(
    test_data.index,
    moving_average_predictions,
    label="Moving Average Forecast"
)

plt.plot(
    test_data.index,
    exponential_predictions,
    label="Exponential Smoothing Forecast"
)

plt.title("Actual vs Forecasted EV Charging Demand")
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.legend()

plt.tight_layout()

plt.savefig(
    "reports/time_series/charts/forecast_comparison.png"
)

plt.close()


# --------------------------------------------------
# 16. COMPLETION
# --------------------------------------------------

print("\n======================================")
print("TIME SERIES FORECASTING COMPLETED!")
print("======================================")

print("\nGenerated files:")
print("1. data/processed/demand_hourly.csv")
print("2. data/processed/demand_daily.csv")
print("3. reports/time_series/forecast_results.csv")
print("4. reports/time_series/forecast_predictions.csv")
print("5. reports/time_series/stationarity_result.txt")
print("6. reports/time_series/charts/")
