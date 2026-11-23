# Exploratory Data Analysis – EV Charging Demand Prediction

## 1. Dataset Overview
The dataset contains 72,826 charging sessions, 14 charging locations, two charger types and two company codes according to the current EDA results.

The average charging demand is 17.44, with a median of 14.1 and a maximum of 97.

## 2. Charging Usage Patterns
Public areas and apartments have the highest number of charging sessions. Public areas recorded 14,076 sessions, followed by apartments with 14,036 sessions.

## 3. Charger Type Analysis
Charger Type 0 accounts for approximately 79.5% of charging sessions. Charger Type 1 has a higher average demand of 18.95 compared with 17.05 for Type 0.

## 4. Company Analysis
Company code 1 recorded 45,709 sessions, while Company code 0 recorded 27,117 sessions. Company code 1 also showed a higher average demand.

## 5. Time-Based Analysis
Charging activity is highest at 18:00, with 6,506 sessions. The highest average demand occurs at 23:00, reaching 27.71.

August 2022 recorded the highest monthly charging activity, with 8,083 sessions.

## 6. Demand and Duration Analysis
The average charging duration calculated from timestamps is 155.54 minutes. The average demand is 17.44.

Potential outliers were identified using the Interquartile Range (IQR) method.

## 7. Important Relationships
Recorded duration has a correlation of 0.657 with demand, while calculated duration has a correlation of 0.604.

These relationships will be investigated further during feature engineering and model development.

## 8. Conclusion
EDA identified meaningful charging patterns across locations, charger types, companies and time periods. These findings provide a foundation for feature engineering, regression, classification and time-series forecasting.