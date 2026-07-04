import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load Dataset

df = pd.read_csv("data/smart-city-traffic-patterns/train_aWnotuB.csv")

# Date Conversion

df["DateTime"] = pd.to_datetime(df["DateTime"])

# Feature Engineering

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek

# Features

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

# Target

y = df["Vehicles"]

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Hyperparameter Grid

param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [10, 20],
    "min_samples_split": [2, 5]
}

rf = RandomForestRegressor(
    random_state=42
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

print("Training...")

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

print("\nBest Parameters")

print(grid_search.best_params_)

# Predictions

predictions = best_model.predict(X_test)

# Metrics

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\nImproved Model Results")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# Save Model

joblib.dump(
    best_model,
    "models/improved_random_forest.pkl"
)

print("\nImproved Model Saved!")
