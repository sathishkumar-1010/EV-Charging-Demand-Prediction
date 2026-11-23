
import pandas as pd
from pathlib import Path

# 1. Load dataset
project_path = Path(__file__).resolve().parent.parent

file_path = (
    project_path / "data" / "interim" /
    "charging_cleaned.csv"
)

df = pd.read_csv(file_path)

# 2. Convert datetime columns
df["StartDatetime"] = pd.to_datetime(df["StartDatetime"])
df["EndDatetime"] = pd.to_datetime(df["EndDatetime"])

# 3. Calculate duration from timestamps
df["CalculatedDuration"] = (
    df["EndDatetime"] - df["StartDatetime"]
).dt.total_seconds() / 60

# 4. Separate valid and invalid timestamps
valid_records = df[
    df["EndDatetime"] >= df["StartDatetime"]
].copy()

invalid_records = df[
    df["EndDatetime"] < df["StartDatetime"]
].copy()

# 5. Find zero-duration records
zero_duration = valid_records[
    valid_records["Duration"] <= 0
]

# 6. Find duration mismatches
valid_records["DurationDifference"] = (
    valid_records["Duration"] -
    valid_records["CalculatedDuration"]
).abs()

mismatches = valid_records[
    valid_records["DurationDifference"] > 1
]

# 7. Display results
print("\n DURATION INVESTIGATION ")

print("Total records:", len(df))
print("Valid timestamp records:", len(valid_records))
print("Invalid timestamp records:", len(invalid_records))

print("\nZero/negative recorded duration with valid timestamps:",
      len(zero_duration))

print("Duration mismatches above 1 minute:", len(mismatches))

# 8. Difference distribution
print("\n DIFFERENCE STATISTICS ")

print(
    valid_records["DurationDifference"].describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

# 9. Show large mismatches
print("\n LARGE MISMATCHES ")

large_mismatches = valid_records[
    valid_records["DurationDifference"] > 60
]

print("Differences above 60 minutes:", len(large_mismatches))

print(
    large_mismatches[
        ["StartDatetime", "EndDatetime",
         "Duration", "CalculatedDuration",
         "DurationDifference"]
    ].head(15).to_string(index=False)
)

# 10. Show extreme calculated durations
print("\n EXTREME DURATIONS ")

print(
    valid_records.nlargest(10, "CalculatedDuration")[
        ["StartDatetime", "EndDatetime",
         "Duration", "CalculatedDuration"]
    ].to_string(index=False)
)

print("\nDuration investigation completed!")
