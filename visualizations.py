import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/smart-city-traffic-patterns/train_aWnotuB.csv")

# Traffic Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Vehicles"], bins=30)
plt.title("Traffic Distribution")
plt.xlabel("Vehicles")
plt.ylabel("Frequency")
plt.savefig("images/traffic_distribution.png")
plt.show()

# Convert DateTime
df["DateTime"] = pd.to_datetime(df["DateTime"])

df["Hour"] = df["DateTime"].dt.hour

# Average Traffic by Hour
hourly = df.groupby("Hour")["Vehicles"].mean()

plt.figure(figsize=(10,5))
hourly.plot()
plt.title("Average Traffic by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Vehicles")
plt.grid()
plt.savefig("images/hourly_traffic.png")
plt.show()

# Traffic by Junction
plt.figure(figsize=(8,5))
sns.boxplot(x="Junction", y="Vehicles", data=df)
plt.title("Traffic by Junction")
plt.savefig("images/junction_traffic.png")
plt.show()
