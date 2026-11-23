
import pandas as pd
from pathlib import Path

# Locate cleaned dataset
project_path = Path(__file__).resolve().parent.parent
file_path = project_path / "data" / "interim" / "charging_cleaned.csv"

df = pd.read_csv(file_path)

# Convert timestamps
df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
df["EndDatetime"] = pd.to_datetime(df["EndDatetime"])

# Identify invalid records
invalid_duration = df[df["Duration"] <= 0]

invalid_timestamps = df[
    df["EndDatetime"] < df["StartDatetime"]
]

# Find records having both problems
both_issues = df[
    (df["Duration"] <= 0) &
    (df["EndDatetime"] < df["StartDatetime"])
]

# Calculate actual duration using timestamps
df["CalculatedDuration"] = (
    df["EndDatetime"] - df["StartDatetime"]
).dt.total_seconds() / 60

print("\n INVALID DURATION RECORDS ")
print("Total:", len(invalid_duration))
print(invalid_duration[
    ["StartDatetime", "EndDatetime", "Duration", "Demand"]
].head(10).to_string(index=False))

print("\n INVALID TIMESTAMP RECORDS ")
print("Total:", len(invalid_timestamps))
print(invalid_timestamps[
    ["StartDatetime", "EndDatetime", "Duration", "Demand"]
].to_string(index=False))

print("\n BOTH ISSUES ")
print("Total:", len(both_issues))

print("\n DURATION COMPARISON ")
print(df[
    ["Duration", "CalculatedDuration"]
].describe())

# Compare recorded and calculated duration
valid_order = df["CalculatedDuration"] >= 0

duration_mismatch = df[
    valid_order &
    ((df["Duration"] - df["CalculatedDuration"]).abs() > 1)
]

print("\n DURATION MISMATCH ")
print("Records with difference above 1 minute:",
      len(duration_mismatch))

print(duration_mismatch[
    ["StartDatetime", "EndDatetime",
     "Duration", "CalculatedDuration"]
].head(10).to_string(index=False))
