
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# =====================================================
# 1. PROJECT PATHS AND SETTINGS
# =====================================================

project_path = Path(__file__).resolve().parent.parent

file_path = (
    project_path / "data" / "processed" /
    "charging_final.csv"
)

report_path = project_path / "reports" / "eda"
chart_path = report_path / "charts"
summary_path = report_path / "summaries"

chart_path.mkdir(parents=True, exist_ok=True)
summary_path.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

# =====================================================
# 2. LOAD DATASET
# =====================================================

df = pd.read_csv(file_path)

df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
df["EndDatetime"] = pd.to_datetime(df["EndDatetime"])

# Extract time features for EDA
df["Hour"] = df["StartDatetime"].dt.hour
df["DayOfWeek"] = df["StartDatetime"].dt.dayofweek
df["Month"] = df["StartDatetime"].dt.month
df["MonthName"] = df["StartDatetime"].dt.strftime("%b")
df["Date"] = df["StartDatetime"].dt.date

df["DayName"] = df["StartDatetime"].dt.day_name()

df["DayType"] = np.where(
    df["DayOfWeek"] >= 5,
    "Weekend",
    "Weekday"
)

df["TimeOfDay"] = pd.cut(
    df["Hour"],
    bins=[-1, 5, 11, 16, 20, 23],
    labels=[
        "Night",
        "Morning",
        "Afternoon",
        "Evening",
        "Late Evening"
    ]
)

print("\n===================================")
print("EV CHARGING DEMAND - EDA ANALYSIS")
print("===================================")

print("\nDataset Shape:", df.shape)
print("Total Charging Sessions:", len(df))
print("Unique Locations:", df["Location"].nunique())
print("Unique Charger Types:", df["ChargerType"].nunique())
print("Unique Companies:", df["ChargerCompany"].nunique())

# =====================================================
# 3. OVERALL DATASET SUMMARY
# =====================================================

print("\n========== OVERALL SUMMARY ==========")

overall_summary = pd.DataFrame({
    "Metric": [
        "Total Sessions",
        "Average Demand",
        "Minimum Demand",
        "Maximum Demand",
        "Median Demand",
        "Average Duration",
        "Median Duration",
        "Total Locations",
        "Total Charger Types",
        "Total Companies"
    ],
    "Value": [
        len(df),
        df["Demand"].mean(),
        df["Demand"].min(),
        df["Demand"].max(),
        df["Demand"].median(),
        df["Duration"].mean(),
        df["Duration"].median(),
        df["Location"].nunique(),
        df["ChargerType"].nunique(),
        df["ChargerCompany"].nunique()
    ]
})

print(overall_summary.to_string(index=False))

overall_summary.to_csv(
    summary_path / "overall_summary.csv",
    index=False
)

# =====================================================
# 4. CHARGING DEMAND DISTRIBUTION
# =====================================================

print("\n========== DEMAND DISTRIBUTION ==========")

print(df["Demand"].describe())

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Demand",
    bins=40,
    kde=True
)

plt.title("Distribution of EV Charging Demand")
plt.xlabel("Demand")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(chart_path / "01_demand_distribution.png")
plt.close()

# Demand boxplot
plt.figure(figsize=(10, 5))

sns.boxplot(x=df["Demand"])

plt.title("EV Charging Demand - Boxplot")
plt.xlabel("Demand")
plt.tight_layout()

plt.savefig(chart_path / "02_demand_boxplot.png")
plt.close()

# =====================================================
# 5. LOCATION-WISE USAGE ANALYSIS
# =====================================================

print("\n========== LOCATION-WISE USAGE ==========")

location_usage = (
    df.groupby("Location")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="TotalSessions")
)

print(location_usage.to_string(index=False))

location_usage.to_csv(
    summary_path / "location_usage.csv",
    index=False
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=location_usage,
    x="TotalSessions",
    y="Location"
)

plt.title("Charging Sessions by Location")
plt.xlabel("Total Charging Sessions")
plt.ylabel("Location")
plt.tight_layout()

plt.savefig(chart_path / "03_location_usage.png")
plt.close()

# =====================================================
# 6. LOCATION-WISE DEMAND ANALYSIS
# =====================================================

print("\n========== LOCATION-WISE DEMAND ==========")

location_demand = (
    df.groupby("Location")
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum"),
        MedianDemand=("Demand", "median")
    )
    .sort_values("AverageDemand", ascending=False)
    .reset_index()
)

print(location_demand.to_string(index=False))

location_demand.to_csv(
    summary_path / "location_demand.csv",
    index=False
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=location_demand,
    x="AverageDemand",
    y="Location"
)

plt.title("Average Charging Demand by Location")
plt.xlabel("Average Demand")
plt.ylabel("Location")
plt.tight_layout()

plt.savefig(chart_path / "04_location_demand.png")
plt.close()

# =====================================================
# 7. CHARGER TYPE USAGE ANALYSIS
# =====================================================

print("\n========== CHARGER TYPE USAGE ==========")

charger_usage = (
    df.groupby("ChargerType")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="TotalSessions")
)

print(charger_usage.to_string(index=False))

charger_usage.to_csv(
    summary_path / "charger_type_usage.csv",
    index=False
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=charger_usage,
    x="ChargerType",
    y="TotalSessions"
)

plt.title("Charging Sessions by Charger Type")
plt.xlabel("Charger Type Code")
plt.ylabel("Total Sessions")
plt.tight_layout()

plt.savefig(chart_path / "05_charger_type_usage.png")
plt.close()

# Charger type distribution
plt.figure(figsize=(7, 7))

plt.pie(
    charger_usage["TotalSessions"],
    labels=charger_usage["ChargerType"].astype(str),
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Charger Type Usage Distribution")
plt.tight_layout()

plt.savefig(chart_path / "06_charger_type_distribution.png")
plt.close()

# =====================================================
# 8. CHARGER TYPE DEMAND ANALYSIS
# =====================================================

print("\n========== CHARGER TYPE DEMAND ==========")

charger_demand = (
    df.groupby("ChargerType")
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum"),
        MedianDemand=("Demand", "median")
    )
    .reset_index()
)

print(charger_demand.to_string(index=False))

charger_demand.to_csv(
    summary_path / "charger_type_demand.csv",
    index=False
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=charger_demand,
    x="ChargerType",
    y="AverageDemand"
)

plt.title("Average Demand by Charger Type")
plt.xlabel("Charger Type Code")
plt.ylabel("Average Demand")
plt.tight_layout()

plt.savefig(chart_path / "07_charger_type_demand.png")
plt.close()

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="ChargerType",
    y="Demand"
)

plt.title("Demand Distribution by Charger Type")
plt.xlabel("Charger Type Code")
plt.ylabel("Demand")
plt.tight_layout()

plt.savefig(chart_path / "08_charger_type_demand_boxplot.png")
plt.close()

# =====================================================
# 9. COMPANY-WISE USAGE ANALYSIS
# =====================================================

print("\n========== COMPANY-WISE USAGE ==========")

company_usage = (
    df.groupby("ChargerCompany")
    .size()
    .reset_index(name="TotalSessions")
)

print(company_usage.to_string(index=False))

company_usage.to_csv(
    summary_path / "company_usage.csv",
    index=False
)

if df["ChargerCompany"].nunique() > 1:

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=company_usage,
        x="ChargerCompany",
        y="TotalSessions"
    )

    plt.title("Charging Sessions by Company")
    plt.xlabel("Company Code")
    plt.ylabel("Total Sessions")
    plt.tight_layout()

    plt.savefig(chart_path / "09_company_usage.png")
    plt.close()

else:
    print(
        "Only one company is available. "
        "Company comparison is not possible."
    )

# =====================================================
# 10. COMPANY-WISE DEMAND ANALYSIS
# =====================================================

print("\n========== COMPANY-WISE DEMAND ==========")

company_demand = (
    df.groupby("ChargerCompany")
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum"),
        MedianDemand=("Demand", "median")
    )
    .reset_index()
)

print(company_demand.to_string(index=False))

company_demand.to_csv(
    summary_path / "company_demand.csv",
    index=False
)

# =====================================================
# 11. HOURLY CHARGING PATTERN
# =====================================================

print("\n========== HOURLY CHARGING PATTERN ==========")

hourly_analysis = (
    df.groupby("Hour")
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum")
    )
    .reindex(range(24))
    .reset_index()
)

print(hourly_analysis.to_string(index=False))

hourly_analysis.to_csv(
    summary_path / "hourly_analysis.csv",
    index=False
)

plt.figure(figsize=(12, 5))

sns.lineplot(
    data=hourly_analysis,
    x="Hour",
    y="TotalSessions",
    marker="o"
)

plt.title("Hourly EV Charging Sessions")
plt.xlabel("Hour of Day")
plt.ylabel("Total Sessions")
plt.xticks(range(24))
plt.tight_layout()

plt.savefig(chart_path / "10_hourly_sessions.png")
plt.close()

plt.figure(figsize=(12, 5))

sns.lineplot(
    data=hourly_analysis,
    x="Hour",
    y="AverageDemand",
    marker="o"
)

plt.title("Average Charging Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Demand")
plt.xticks(range(24))
plt.tight_layout()

plt.savefig(chart_path / "11_hourly_demand.png")
plt.close()

# =====================================================
# 12. DAY-WISE CHARGING ANALYSIS
# =====================================================

print("\n========== DAY-WISE ANALYSIS ==========")

day_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

daily_analysis = (
    df.groupby("DayName")
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum")
    )
    .reindex(day_order)
    .reset_index()
)

print(daily_analysis.to_string(index=False))

daily_analysis.to_csv(
    summary_path / "daily_analysis.csv",
    index=False
)

plt.figure(figsize=(11, 5))

sns.barplot(
    data=daily_analysis,
    x="DayName",
    y="TotalSessions"
)

plt.title("Charging Sessions by Day of Week")
plt.xlabel("Day")
plt.ylabel("Total Sessions")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(chart_path / "12_daily_sessions.png")
plt.close()

# =====================================================
# 13. WEEKDAY VS WEEKEND ANALYSIS
# =====================================================

print("\n========== WEEKDAY VS WEEKEND ==========")

weekend_analysis = (
    df.groupby("DayType")
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum")
    )
    .reset_index()
)

print(weekend_analysis.to_string(index=False))

weekend_analysis.to_csv(
    summary_path / "weekday_weekend_analysis.csv",
    index=False
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=weekend_analysis,
    x="DayType",
    y="AverageDemand"
)

plt.title("Average Demand: Weekday vs Weekend")
plt.xlabel("Day Type")
plt.ylabel("Average Demand")
plt.tight_layout()

plt.savefig(chart_path / "13_weekday_weekend.png")
plt.close()

# =====================================================
# 14. MONTHLY CHARGING TREND
# =====================================================

print("\n========== MONTHLY ANALYSIS ==========")

monthly_analysis = (
    df.groupby(df["StartDatetime"].dt.to_period("M"))
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum")
    )
)

monthly_analysis.index = monthly_analysis.index.astype(str)

monthly_analysis = monthly_analysis.reset_index(
    names="Month"
)

print(monthly_analysis.to_string(index=False))

monthly_analysis.to_csv(
    summary_path / "monthly_analysis.csv",
    index=False
)

plt.figure(figsize=(13, 5))

sns.lineplot(
    data=monthly_analysis,
    x="Month",
    y="TotalSessions",
    marker="o"
)

plt.title("Monthly EV Charging Sessions")
plt.xlabel("Month")
plt.ylabel("Total Sessions")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(chart_path / "14_monthly_sessions.png")
plt.close()

plt.figure(figsize=(13, 5))

sns.lineplot(
    data=monthly_analysis,
    x="Month",
    y="AverageDemand",
    marker="o"
)

plt.title("Monthly Average Charging Demand")
plt.xlabel("Month")
plt.ylabel("Average Demand")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(chart_path / "15_monthly_demand.png")
plt.close()

# =====================================================
# 15. TIME-OF-DAY ANALYSIS
# =====================================================

print("\n========== TIME-OF-DAY ANALYSIS ==========")

time_analysis = (
    df.groupby("TimeOfDay", observed=False)
    .agg(
        TotalSessions=("Demand", "count"),
        AverageDemand=("Demand", "mean"),
        TotalDemand=("Demand", "sum")
    )
    .reset_index()
)

print(time_analysis.to_string(index=False))

time_analysis.to_csv(
    summary_path / "time_of_day_analysis.csv",
    index=False
)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=time_analysis,
    x="TimeOfDay",
    y="TotalSessions"
)

plt.title("Charging Sessions by Time of Day")
plt.xlabel("Time Period")
plt.ylabel("Total Sessions")
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(chart_path / "16_time_of_day.png")
plt.close()

# =====================================================
# 16. DURATION ANALYSIS
# =====================================================

print("\n========== DURATION ANALYSIS ==========")

print(df["Duration"].describe())

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Duration",
    bins=50,
    kde=True
)

plt.title("Charging Duration Distribution")
plt.xlabel("Calculated Duration (Minutes)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(chart_path / "17_duration_distribution.png")
plt.close()

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["Duration"])

plt.title("Charging Duration Outlier Analysis")
plt.xlabel("Duration (Minutes)")
plt.tight_layout()

plt.savefig(chart_path / "18_duration_boxplot.png")
plt.close()

# =====================================================
# 17. OUTLIER DETECTION USING IQR
# =====================================================

print("\n========== OUTLIER DETECTION ==========")

outlier_columns = ["Demand", "Duration", "RecordedDuration"]

outlier_results = []

for column in outlier_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    result = {
        "Column": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LowerLimit": lower_limit,
        "UpperLimit": upper_limit,
        "OutlierCount": len(outliers),
        "OutlierPercentage": len(outliers) / len(df) * 100
    }

    outlier_results.append(result)

    print(f"\n{column}")
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Outliers:", len(outliers))

outlier_df = pd.DataFrame(outlier_results)

outlier_df.to_csv(
    summary_path / "outlier_analysis.csv",
    index=False
)

# =====================================================
# 18. CORRELATION ANALYSIS
# =====================================================

print("\n========== CORRELATION ANALYSIS ==========")

correlation_columns = [
    "Demand",
    "Duration",
    "RecordedDuration",
    "CalculatedDuration"
]

correlation = df[correlation_columns].corr()

print("\nCorrelation with Demand:")
print(
    correlation["Demand"].sort_values(
        ascending=False
    )
)

correlation.to_csv(
    summary_path / "correlation_matrix.csv"
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig(chart_path / "19_correlation_heatmap.png")
plt.close()

# =====================================================
# 19. IMPORTANT RELATIONSHIPS
# =====================================================

print("\n========== IMPORTANT RELATIONSHIPS ==========")

# Demand vs Duration
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Duration",
    y="Demand",
    alpha=0.3
)

plt.title("Relationship Between Duration and Demand")
plt.xlabel("Calculated Duration (Minutes)")
plt.ylabel("Demand")
plt.tight_layout()

plt.savefig(chart_path / "20_duration_vs_demand.png")
plt.close()

# Demand vs Recorded Duration
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="RecordedDuration",
    y="Demand",
    alpha=0.3
)

plt.title("Relationship Between Recorded Duration and Demand")
plt.xlabel("Recorded Duration (Minutes)")
plt.ylabel("Demand")
plt.tight_layout()

plt.savefig(chart_path / "21_recorded_duration_vs_demand.png")
plt.close()

# Demand vs Hour
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Hour",
    y="Demand",
    alpha=0.3
)

plt.title("Relationship Between Charging Hour and Demand")
plt.xlabel("Hour")
plt.ylabel("Demand")
plt.tight_layout()

plt.savefig(chart_path / "22_hour_vs_demand.png")
plt.close()

# =====================================================
# 20. LOCATION AND CHARGER TYPE RELATIONSHIP
# =====================================================

print("\n========== LOCATION VS CHARGER TYPE ==========")

location_charger = pd.crosstab(
    df["Location"],
    df["ChargerType"]
)

print(location_charger)

location_charger.to_csv(
    summary_path / "location_charger_type.csv"
)

plt.figure(figsize=(11, 7))

sns.heatmap(
    location_charger,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Charging Sessions: Location vs Charger Type")
plt.xlabel("Charger Type Code")
plt.ylabel("Location")
plt.tight_layout()

plt.savefig(chart_path / "23_location_charger_heatmap.png")
plt.close()

# =====================================================
# 21. HIGH-DEMAND PATTERN ANALYSIS
# =====================================================

print("\n========== HIGH-DEMAND ANALYSIS ==========")

# Median-based threshold as specified in project document
demand_threshold = df["Demand"].median()

df["DemandCategory"] = np.where(
    df["Demand"] >= demand_threshold,
    "High Demand",
    "Normal Demand"
)

print("Demand Threshold:", demand_threshold)

print(df["DemandCategory"].value_counts())

high_demand_location = (
    df[df["DemandCategory"] == "High Demand"]
    .groupby("Location")
    .size()
    .sort_values(ascending=False)
    .reset_index(name="HighDemandSessions")
)

high_demand_location.to_csv(
    summary_path / "high_demand_location.csv",
    index=False
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=high_demand_location,
    x="HighDemandSessions",
    y="Location"
)

plt.title("High-Demand Sessions by Location")
plt.xlabel("High-Demand Sessions")
plt.ylabel("Location")
plt.tight_layout()

plt.savefig(chart_path / "24_high_demand_location.png")
plt.close()

# =====================================================
# 22. SAVE HIGH-DEMAND TIME ANALYSIS
# =====================================================

high_demand_time = (
    df[df["DemandCategory"] == "High Demand"]
    .groupby("Hour")
    .size()
    .reindex(range(24), fill_value=0)
    .reset_index(name="HighDemandSessions")
)

print("\nHigh-demand sessions by hour:")
print(high_demand_time.to_string(index=False))

high_demand_time.to_csv(
    summary_path / "high_demand_by_hour.csv",
    index=False
)

plt.figure(figsize=(12, 5))

sns.lineplot(
    data=high_demand_time,
    x="Hour",
    y="HighDemandSessions",
    marker="o"
)

plt.title("High-Demand Charging Sessions by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("High-Demand Sessions")
plt.xticks(range(24))
plt.tight_layout()

plt.savefig(chart_path / "25_high_demand_by_hour.png")
plt.close()

# =====================================================
# 23. FINAL EDA SUMMARY
# =====================================================

print("\n===================================")
print("EDA ANALYSIS COMPLETED SUCCESSFULLY")
print("===================================")

print("Total Records:", len(df))
print("Total Locations:", df["Location"].nunique())
print("Total Charger Types:", df["ChargerType"].nunique())
print("Total Companies:", df["ChargerCompany"].nunique())

print("Average Demand:", df["Demand"].mean())
print("Median Demand:", df["Demand"].median())
print("Maximum Demand:", df["Demand"].max())

print("\nCharts saved in:", chart_path)
print("Summaries saved in:", summary_path)

print("\nAll EDA analyses completed!")
