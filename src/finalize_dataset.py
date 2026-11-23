
import pandas as pd
from pathlib import Path
# 1. PROJECT PATHS

project_path = Path(__file__).resolve().parent.parent

input_path = (
    project_path / "data" / "interim" /
    "charging_cleaned.csv"
)

output_folder = project_path / "data" / "processed"
output_folder.mkdir(parents=True, exist_ok=True)

output_path = output_folder / "charging_final.csv"

# 2. LOAD DATASET

df = pd.read_csv(input_path)

df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
df["EndDatetime"] = pd.to_datetime(df["EndDatetime"])

print("Original shape:", df.shape)

# 3. REMOVE INVALID TIMESTAMP RECORDS

invalid_timestamp = (
    df["EndDatetime"] < df["StartDatetime"]
)

print("Invalid timestamp records:", invalid_timestamp.sum())

df = df.loc[~invalid_timestamp].copy()

# 4. PRESERVE ORIGINAL DURATION

df.rename(
    columns={"Duration": "RecordedDuration"},
    inplace=True
)

# 5. CALCULATE DURATION FROM TIMESTAMPS

df["CalculatedDuration"] = (
    df["EndDatetime"] - df["StartDatetime"]
).dt.total_seconds() / 60

# 6. CREATE DURATION QUALITY FLAG

df["DurationMismatch"] = (
    (df["RecordedDuration"] -
     df["CalculatedDuration"]).abs() > 1
)

# Use calculated duration as primary duration feature
df["Duration"] = df["CalculatedDuration"]

# 7. CHECK DATA QUALITY

print("\n FINAL DATA QUALITY ")

print("Final records:", len(df))

print(
    "Zero calculated duration:",
    (df["Duration"] == 0).sum()
)

print(
    "Negative calculated duration:",
    (df["Duration"] < 0).sum()
)

print(
    "Duration mismatch flags:",
    df["DurationMismatch"].sum()
)

print("\nMissing values:")
print(df.isnull().sum())

# 8. SAVE FINAL DATASET

df.to_csv(output_path, index=False)

print("\n FINAL SUMMARY ")

print("Final dataset shape:", df.shape)
print("Saved location:", output_path)

print("\nFinal preprocessing completed!")
