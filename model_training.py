import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib

# Load cleaned dataset

df = pd.read_csv("data/smart-city-traffic-patterns/train_aWnotuB.csv")


# Convert DateTime

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

# Split Data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Models

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

best_model = None
best_r2 = -999

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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

    results.append(
        [name, mae, rmse, r2]
    )

    print("\n", name)

    print("MAE:", mae)

    print("RMSE:", rmse)

    print("R2:", r2)

    if r2 > best_r2:

        best_r2 = r2

        best_model = model

# Save Best Model

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print(
    "\nBest Model Saved Successfully!"
)
