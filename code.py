import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Step 1: Create sample data simulating energy operations
np.random.seed(42)
sample_data = pd.DataFrame({
    'Steam_Temperature': np.random.uniform(200, 400, 1000),  # in Celsius
    'Steam_Pressure': np.random.uniform(30, 100, 1000),  # in bar
    'Fuel_Mix': np.random.uniform(0.1, 0.9, 1000),  # Ratio of fuel types
    'Efficiency': np.random.uniform(0.7, 1.0, 1000),  # Efficiency ratio
    'Emissions': np.random.choice([0, 1], size=1000, p=[0.7, 0.3])  # Emission binary classes
})

# Save sample data for reuse
sample_data_path = "sample_energy_data.csv"
sample_data.to_csv(sample_data_path, index=False)

# Step 2: Data Preprocessing
scaler = StandardScaler()
numeric_columns = ['Steam_Temperature', 'Steam_Pressure', 'Fuel_Mix']
sample_data[numeric_columns] = scaler.fit_transform(sample_data[numeric_columns])

# Save the scaler for future use
scaler_path = "scaler.joblib"
joblib.dump(scaler, scaler_path)

# Step 3: Split the data for model training and testing
X = sample_data[numeric_columns]
y = sample_data['Efficiency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Heat Rate Optimization Model (Regression)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)
# Save the trained model
regressor_path = "heat_rate_model.joblib"
joblib.dump(regressor, regressor_path)

# Evaluate the Heat Rate Optimization Model
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Step 5: Train the Anomaly Detection Model (Isolation Forest)
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
anomaly_detector.fit(X_train)
# Save the trained model
anomaly_model_path = "anomaly_detector.joblib"
joblib.dump(anomaly_detector, anomaly_model_path)

# Step 6: Train the Emission Classification Model (Classification)
X_class = sample_data[numeric_columns]
y_class = sample_data['Emissions']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

classifier = RandomForestRegressor(n_estimators=100, random_state=42)
classifier.fit(X_train_c, y_train_c)
# Save the trained model
classifier_path = "emission_classifier.joblib"
joblib.dump(classifier, classifier_path)

# Evaluate the Emission Classification Model
y_pred_c = np.round(classifier.predict(X_test_c))
f1 = f1_score(y_test_c, y_pred_c)

# Step 7: Save Results Summary
results_summary = {
    "Heat Rate Optimization (MAE)": mae,
    "Emission Prediction (F1 Score)": f1
}
results_summary_path = "results_summary.json"
with open(results_summary_path, "w") as f:
    json.dump(results_summary, f)

# Visualization: Model Performance Metrics
metrics = {
    "Heat Rate Optimization (MAE)": mae,
    "Emission Prediction (F1 Score)": f1
}
labels = list(metrics.keys())
values = list(metrics.values())

# Create a bar chart for the performance metrics
plt.figure(figsize=(10, 6))
plt.barh(labels, values, color=['skyblue', 'orange'])
plt.xlabel('Metrics Value')
plt.title('Model Performance Metrics')
plt.xlim(0, max(values) * 1.2)

# Annotate the bars
for i, v in enumerate(values):
    plt.text(v + 0.02, i, f"{v:.3f}", va='center')

# Save the plot
plot_path = "model_performance_metrics.png"
plt.savefig(plot_path)
plt.show()

# Save deployment paths for reference
output_files = {
    "Sample Data Path": sample_data_path,
    "Scaler Path": scaler_path,
    "Heat Rate Model": regressor_path,
    "Anomaly Detector Model": anomaly_model_path,
    "Emission Classifier Model": classifier_path,
    "Results Summary": results_summary_path,
    "Performance Metrics Visualization": plot_path
}
output_files_path = "deployment_paths.json"
with open(output_files_path, "w") as f:
    json.dump(output_files, f)

# Output all files for user reference
output_files
