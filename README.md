# upskillcampus
Machine Learning project for Smart City Traffic Forecasting using Python, Scikit-Learn, and Random Forest Regression with traffic pattern analysis and forecasting.

# Smart City Traffic Forecasting

## 📌 Project Overview
Smart City Traffic Forecasting is a Machine Learning project developed to analyze historical traffic patterns and predict future traffic volumes at different city junctions. The project uses traffic data and Machine Learning algorithms to forecast vehicle counts, helping in better traffic management and smart city planning.

---

## 🎯 Project Objectives
- Analyze historical traffic data.
- Identify traffic patterns and trends.
- Build predictive machine learning models.
- Compare model performances.
- Forecast future traffic volumes.
- Generate visual insights for traffic management.

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- VS Code
- Git & GitHub

---

## 📂 Dataset Information
The dataset contains traffic information collected from multiple city junctions.

### Features
- **DateTime** – Timestamp of traffic observation
- **Junction** – Junction identifier
- **Vehicles** – Number of vehicles observed
- **ID** – Unique observation ID

**Dataset Size:** 48,120 records

---

## 🚀 Project Workflow

### Week 1 – Data Preprocessing and EDA
- Dataset loading
- Data cleaning
- Missing value analysis
- DateTime conversion
- Exploratory Data Analysis (EDA)
- Traffic pattern visualization

### Week 2 – Model Development
Implemented:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

Evaluation Metrics:
- MAE
- RMSE
- R² Score

### Week 3 – Model Optimization
- Hyperparameter tuning using GridSearchCV
- Feature importance analysis
- Actual vs Predicted visualization
- Improved Random Forest model

### Week 4 – Traffic Forecasting
- Future traffic prediction
- Test dataset forecasting
- Final forecast visualization
- Project documentation and deployment

---

## 📊 Model Performance

| Model | R² Score |
|-------|-----------|
| Linear Regression | 0.5997 |
| Decision Tree | 0.9436 |
| Random Forest | 0.9690 |

### 🏆 Best Model
**Random Forest Regressor** achieved the highest performance with an **R² Score of 0.9690**.

---

## 📈 Generated Visualizations
- Traffic Distribution Analysis
- Average Traffic by Hour
- Traffic Distribution by Junction
- Model Performance Comparison
- Feature Importance Analysis
- Actual vs Predicted Traffic
- Final Forecast Visualization

---

## 📁 Project Structure

```text
upskillcampus
│
├── data
     ├──train_aWnotuB.csv
     ├──datasets_8494_11879_test_BdBKkAj.csv
     ├──traffic_cleaned.csv

├── images
    ├──traffic_distribution.png
    ├──hourly_traffic.png
    ├──unction_traffic.png
    ├──model_comparison.png
    ├──feature_importance.png
    ├──actual_vs_predicted.png
    ├──final_forecast.png

├── models/
     ├──best_model.pkl
     ├── improved_random_forest.pkl
 
├── reports/
     ├── final_predictions.csk

├── src/
     ├──eda.py
     ├──feature_importance.py
     ├──final_prediction.py
     ├──final_visualization.py
     ├──model_comparision.py
     ├──model_improvement.py
     ├──model_training.py
     ├──  prediction_visualization.py
     ├── visualizations.py

├── README.md

├── requirements.txt

├── Reports/
     ├── Week-1.pdf
     ├── Week-2.pdf
     ├── Week-3.pdf
     ├── Week-4.pdf
     └── Karri_Himabindu_Smart_City_Traffic_Forecasting_Report.pdf
```

---

## ⚙️ Installation

### Clone Repository
```bash
git clone https://github.com/your-username/upskillcampus.git
cd upskillcampus
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Project
```bash
python src/final_prediction.py
python src/final_visualization.py
```

---

## 📌 Results
- Successfully analyzed traffic patterns.
- Built and compared multiple regression models.
- Achieved high prediction accuracy using Random Forest.
- Generated future traffic forecasts on unseen data.
- Created visual insights for smart city traffic management.

---

## 🔮 Future Scope
- Real-time traffic data integration
- Deep Learning (LSTM) based forecasting
- Smart traffic signal optimization
- Interactive web dashboard deployment
- Real-time traffic monitoring system

---

## 👩‍💻 Author

**Karri Himabindu**  

**Internship Project:** Internship in Data Science & Machine Learning  
**Project Title:** Smart City Traffic Forecasting
