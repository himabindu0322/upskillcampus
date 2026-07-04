import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Load Data

df = pd.read_csv(
    "data/traffic_cleaned.csv"
)

df["DateTime"] = pd.to_datetime(
    df["DateTime"]
)

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek

X = df[
    [
        "Junction",
        "Year",
        "Month",
        "Day",
        "Hour",
        "DayOfWeek"
    ]
]

y = df["Vehicles"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = joblib.load(
    "models/improved_random_forest.pkl"
)

predictions = model.predict(
    X_test
)

plt.figure(figsize=(10,5))

plt.plot(
    y_test.values[:100],
    label="Actual"
)

plt.plot(
    predictions[:100],
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Traffic"
)

plt.xlabel(
    "Observations"
)

plt.ylabel(
    "Vehicles"
)

plt.legend()

plt.savefig(
    "images/actual_vs_predicted.png"
)

plt.show()

print(
    "Prediction Graph Saved!"
)
