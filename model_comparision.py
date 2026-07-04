import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "R2": [
            0.5997395466492281,  
            0.943607078705784,  
            0.9689627529667615  
        ]
    }
)

plt.figure(figsize=(8,5))

plt.bar(results["Model"], results["R2"])

plt.title("Model Performance Comparison")

plt.xlabel("Models")

plt.ylabel("R² Score")

plt.savefig("images/model_comparison.png")

plt.show()

print("Graph saved successfully!")
