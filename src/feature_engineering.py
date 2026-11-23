
import pandas as pd
import os

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("data/processed/charging_final.csv")

df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"],
    errors="coerce"
)

print("Original dataset shape:", df.shape)

# Sort sessions chronologically
df = df.sort_values(
    "StartDatetime"
).reset_index(drop=True)


# ==========================================
# STEP 2: TIME-BASED FEATURES
# ==========================================

df["Hour"] = df["StartDatetime"].dt.hour
df["Day"] = df["StartDatetime"].dt.day
df["Month"] = df["StartDatetime"].dt.month

# Monday = 0, Sunday = 6
df["DayOfWeek"] = df["StartDatetime"].dt.dayofweek

df["DayName"] = df["StartDatetime"].dt.day_name()

# 1 = Weekend, 0 = Weekday
df["IsWeekend"] = (
    df["DayOfWeek"] >= 5
).astype(int)

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
).astype("string")

print("\nTime-based features created.")


# ==========================================
# STEP 3: HISTORICAL FEATURE FUNCTION
# ==========================================

def create_historical_features(data, group_column):

    # Group sessions by category and exact start time.
    # Sessions with the same timestamp are treated together.
    history = (
        data.groupby(
            [group_column, "StartDatetime"],
            dropna=False
        )
        .agg(
            SessionCount=("Demand", "size"),
            DemandSum=("Demand", "sum")
        )
        .reset_index()
    )

    # Sort each group's timestamps
    history = history.sort_values(
        [group_column, "StartDatetime"]
    )

    # Calculate cumulative previous session count and demand.
    # Subtract the current timestamp group so that
    # simultaneous sessions cannot use each other's demand.
    history["PastSessionCount"] = (
        history.groupby(group_column, dropna=False)[
            "SessionCount"
        ].cumsum()
        - history["SessionCount"]
    )

    history["PastDemandSum"] = (
        history.groupby(group_column, dropna=False)[
            "DemandSum"
        ].cumsum()
        - history["DemandSum"]
    )

    # Calculate previous average demand
    history["PastAvgDemand"] = (
        history["PastDemandSum"]
        / history["PastSessionCount"].replace(0, pd.NA)
    )

    # Keep only the columns needed for merging
    history = history[
        [
            group_column,
            "StartDatetime",
            "PastSessionCount",
            "PastAvgDemand"
        ]
    ]

    # Merge historical information back to every session
    data = data.merge(
        history,
        on=[group_column, "StartDatetime"],
        how="left",
        validate="many_to_one"
    )

    return data


# ==========================================
# STEP 4: CREATE HISTORICAL FEATURES
# ==========================================

# Location history
df = create_historical_features(
    df,
    "Location"
)

df = df.rename(
    columns={
        "PastSessionCount": "Location_PastSessionCount",
        "PastAvgDemand": "Location_PastAvgDemand"
    }
)

# Charger type history
df = create_historical_features(
    df,
    "ChargerType"
)

df = df.rename(
    columns={
        "PastSessionCount": "ChargerType_PastSessionCount",
        "PastAvgDemand": "ChargerType_PastAvgDemand"
    }
)

# Company history
df = create_historical_features(
    df,
    "ChargerCompany"
)

df = df.rename(
    columns={
        "PastSessionCount": "Company_PastSessionCount",
        "PastAvgDemand": "Company_PastAvgDemand"
    }
)

print("\nHistorical features created.")


# ==========================================
# STEP 5: HANDLE SESSIONS WITHOUT HISTORY
# ==========================================

count_columns = [
    "Location_PastSessionCount",
    "ChargerType_PastSessionCount",
    "Company_PastSessionCount"
]

average_columns = [
    "Location_PastAvgDemand",
    "ChargerType_PastAvgDemand",
    "Company_PastAvgDemand"
]

# Create indicators showing whether historical
# information was available for each category.
for count_column, average_column in zip(
    count_columns,
    average_columns
):
    indicator = count_column.replace(
        "PastSessionCount",
        "HasPastHistory"
    )

    df[indicator] = (
        df[count_column] > 0
    ).astype(int)

# No previous sessions means previous count is zero.
df[count_columns] = (
    df[count_columns].fillna(0).astype(int)
)

# No previous average is represented by zero.
# The HasPastHistory columns distinguish this case.
df[average_columns] = (
    df[average_columns].fillna(0)
)


# ==========================================
# STEP 6: PREPARE ML FEATURES
# ==========================================

feature_columns = [
    "Location",
    "ChargerType",
    "ChargerCompany",
    "Hour",
    "Day",
    "Month",
    "DayOfWeek",
    "DayName",
    "IsWeekend",
    "TimeOfDay",
    "Location_PastSessionCount",
    "Location_PastAvgDemand",
    "Location_HasPastHistory",
    "ChargerType_PastSessionCount",
    "ChargerType_PastAvgDemand",
    "ChargerType_HasPastHistory",
    "Company_PastSessionCount",
    "Company_PastAvgDemand",
    "Company_HasPastHistory"
]

# Demand is our prediction target
target_column = "Demand"

ml_df = df[
    ["StartDatetime"] + feature_columns + [target_column]
].copy()


# ==========================================
# STEP 7: CHECK FINAL DATASET
# ==========================================

print("\nFinal ML dataset shape:", ml_df.shape)

print("\nMissing values:")
print(ml_df.isnull().sum())

print("\nFeature columns:")
print(feature_columns)

print("\nTarget column:", target_column)

print("\nSample final records:")
print(ml_df.head(10))


# ==========================================
# STEP 8: SAVE FINAL DATASET
# ==========================================

output_folder = "data/processed"

os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(
    output_folder,
    "charging_ml_dataset.csv"
)

ml_df.to_csv(
    output_path,
    index=False
)

print("\nFinal ML dataset saved successfully!")
print("File location:", output_path)
print("Total records:", len(ml_df))
print("Total input features:", len(feature_columns))
