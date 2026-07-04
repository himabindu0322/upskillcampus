import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/smart-city-traffic-patterns/train_aWnotuB.csv")

print("First 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

# Feature Extraction
df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.day_name()

print("\nUpdated Dataset")
print(df.head())

print("\nStatistical Summary")
print(df.describe())

# Save cleaned dataset
df.to_csv("data/traffic_cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully")
