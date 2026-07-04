import pandas as pd
import joblib

# Load Test Dataset

test_df = pd.read_csv(
    "data/smart-city-traffic-patterns/datasets_8494_11879_test_BdBKkAj.csv"
)

# Convert DateTime

test_df["DateTime"] = pd.to_datetime(
    test_df["DateTime"]
)

# Feature Engineering

test_df["Year"] = test_df["DateTime"].dt.year
test_df["Month"] = test_df["DateTime"].dt.month
test_df["Day"] = test_df["DateTime"].dt.day
test_df["Hour"] = test_df["DateTime"].dt.hour
test_df["DayOfWeek"] = test_df["DateTime"].dt.dayofweek

# Features

X_test = test_df[
    [
        "Junction",
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek"
    ]
]

# Load Best Model

model = joblib.load(
    "models/improved_random_forest.pkl"
)

# Predict

predictions = model.predict(
    X_test
)

# Save Predictions

test_df["Predicted_Vehicles"] = predictions

test_df.to_csv(
    "reports/final_predictions.csv",
    index=False
)

print(
    "Predictions Saved Successfully!"
)
