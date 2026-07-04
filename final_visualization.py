import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "reports/final_predictions.csv"
)

plt.figure(figsize=(12,5))

plt.plot(
    df["Predicted_Vehicles"][:200]
)

plt.title(
    "Traffic Forecast Prediction"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Predicted Vehicles"
)

plt.savefig(
    "images/final_forecast.png"
)

plt.show()

print(
    "Forecast Graph Saved!"
)
