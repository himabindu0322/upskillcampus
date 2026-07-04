import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load(
    "models/improved_random_forest.pkl"
)

features = [
    "Junction",
    "Year",
    "Month",
    "Day",
    "Hour",
    "DayOfWeek"
]

importance = model.feature_importances_

# NEW CODE
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance Ranking:")
print(feature_importance)

# Graph
plt.figure(figsize=(8,5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance Score")

plt.savefig("images/feature_importance.png")

plt.show()

print("Feature Importance Graph Saved!")
