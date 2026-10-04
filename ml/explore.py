import pandas as pd

data = pd.read_csv("data/training_set.csv", parse_dates=["event_timestamp"])

print(data.head())
print(data.describe())


print(
    "=================================================================================================================================================="
)


print("1. Count label 1 overall:")
print(data["label"].mean())


print(
    "=================================================================================================================================================="
)


print("2. Change of rate by month:")
print(data.groupby("event_timestamp")["label"].mean())


print(
    "=================================================================================================================================================="
)


print("3. How much of each feature is missing:")
print(data.isna().mean())


print(
    "=================================================================================================================================================="
)

print("4. Do the features differ between buyers and non buyers")
print(data.groupby("label").mean(numeric_only=True))


print(
    "=================================================================================================================================================="
)
