
import os
import pandas as pd


# ==========================================================
# STEP 1: LOAD FINAL ML DATASET
# ==========================================================

DATA_PATH = "data/processed/charging_ml_dataset.csv"
OUTPUT_FOLDER = "data/processed"

df = pd.read_csv(DATA_PATH)

df["StartDatetime"] = pd.to_datetime(
    df["StartDatetime"],
    errors="coerce"
)

if df["StartDatetime"].isna().any():
    raise ValueError(
        "Missing or invalid StartDatetime values found."
    )

df = df.sort_values(
    "StartDatetime"
).reset_index(drop=True)

print("=" * 60)
print("CHRONOLOGICAL TRAIN-TEST SPLIT")
print("=" * 60)

print("Dataset shape:", df.shape)


# ==========================================================
# STEP 2: DEFINE FEATURES AND TARGET
# ==========================================================

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

target_column = "Demand"

required_columns = (
    feature_columns
    + [target_column, "StartDatetime"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================================
# STEP 3: VALIDATE HISTORICAL FEATURES
# ==========================================================

historical_columns = [
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

if df[historical_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in historical features."
    )

if (df[
    [
        "Location_PastSessionCount",
        "ChargerType_PastSessionCount",
        "Company_PastSessionCount"
    ]
] < 0).any().any():

    raise ValueError(
        "Negative historical session counts detected."
    )

print("\nHistorical feature validation successful.")

print(
    "Historical features will be preserved from "
    "feature_engineering.py."
)


# ==========================================================
# STEP 4: CHRONOLOGICAL TRAIN-TEST SPLIT
# ==========================================================

unique_timestamps = sorted(
    df["StartDatetime"].unique()
)

split_index = int(
    len(unique_timestamps) * 0.80
)

if split_index <= 0 or split_index >= len(unique_timestamps):
    raise ValueError(
        "Invalid chronological split index."
    )

split_timestamp = unique_timestamps[split_index]

train_df = df[
    df["StartDatetime"] < split_timestamp
].copy()

test_df = df[
    df["StartDatetime"] >= split_timestamp
].copy()

print("\nChronological split completed.")

print("Split timestamp:", split_timestamp)


# ==========================================================
# STEP 5: VERIFY HISTORICAL FEATURES ARE PRESERVED
# ==========================================================

# Do NOT replace test historical features with fixed
# training averages or counts.
#
# The feature_engineering.py script already calculates
# historical information using previous sessions only.
#
# This allows earlier test-period sessions to contribute
# to later test-period predictions after their demand
# becomes available.


# ==========================================================
# STEP 6: SEPARATE INPUT FEATURES AND TARGET
# ==========================================================

X_train = train_df[feature_columns].copy()

X_test = test_df[feature_columns].copy()

y_train = train_df[target_column].copy()

y_test = test_df[target_column].copy()


# ==========================================================
# STEP 7: DISPLAY SPLIT INFORMATION
# ==========================================================

print("\nTRAINING DATA")

print("X_train shape:", X_train.shape)

print("y_train shape:", y_train.shape)

print("\nTESTING DATA")

print("X_test shape:", X_test.shape)

print("y_test shape:", y_test.shape)


print("\nTraining period:")

print(train_df["StartDatetime"].min())

print("to")

print(train_df["StartDatetime"].max())


print("\nTesting period:")

print(test_df["StartDatetime"].min())

print("to")

print(test_df["StartDatetime"].max())


# ==========================================================
# STEP 8: VERIFY TIMESTAMP SEPARATION
# ==========================================================

common_timestamps = set(
    train_df["StartDatetime"]
).intersection(
    set(test_df["StartDatetime"])
)

print("\nCommon timestamps:", len(common_timestamps))

if common_timestamps:

    raise ValueError(
        "Timestamp overlap detected between training "
        "and testing data!"
    )

print(
    "SUCCESS: No identical timestamps across "
    "training and testing data."
)


if (
    train_df["StartDatetime"].max()
    >= test_df["StartDatetime"].min()
):

    raise ValueError(
        "Chronological order verification failed!"
    )

print(
    "SUCCESS: Training data comes before testing data."
)


# ==========================================================
# STEP 9: VERIFY FEATURE DIMENSIONS
# ==========================================================

if X_train.shape[1] != 19:

    raise ValueError(
        "Training feature count is not 19."
    )

if X_test.shape[1] != 19:

    raise ValueError(
        "Testing feature count is not 19."
    )

print("\nFeature dimension verification successful.")

print("Number of input features:", X_train.shape[1])


# ==========================================================
# STEP 10: SAVE DATASETS
# ==========================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

train_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "train_dataset.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "test_dataset.csv"
    ),
    index=False
)

X_train.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "X_train.csv"
    ),
    index=False
)

X_test.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "X_test.csv"
    ),
    index=False
)

y_train.to_frame().to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "y_train.csv"
    ),
    index=False
)

y_test.to_frame().to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "y_test.csv"
    ),
    index=False
)


# ==========================================================
# STEP 11: FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)

print("DATA PREPARATION COMPLETED SUCCESSFULLY!")

print("=" * 60)

print("Training records:", len(train_df))

print("Testing records:", len(test_df))

print("Training features:", X_train.shape[1])

print("Testing features:", X_test.shape[1])

print(
    "Historical features preserved from "
    "feature engineering."
)

print(
    "All training and testing files saved successfully!"
)
