
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

train_path = DATA_DIR / "train_dataset.csv"
test_path = DATA_DIR / "test_dataset.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

train["StartDatetime"] = pd.to_datetime(train["StartDatetime"])
test["StartDatetime"] = pd.to_datetime(test["StartDatetime"])

train_times = set(train["StartDatetime"])
test_times = set(test["StartDatetime"])

common_times = train_times.intersection(test_times)

print("TRAINING RECORDS:", len(train))
print("TESTING RECORDS:", len(test))

print("\nTRAINING PERIOD:")
print(train["StartDatetime"].min(), "to", train["StartDatetime"].max())

print("\nTESTING PERIOD:")
print(test["StartDatetime"].min(), "to", test["StartDatetime"].max())

print("\nCOMMON TIMESTAMPS:", len(common_times))

if common_times:
    print("\nWARNING: Identical timestamps exist in both datasets.")
    print("Example common timestamps:")
    print(list(common_times)[:10])
else:
    print("\nSUCCESS: No identical timestamps across train and test.")
