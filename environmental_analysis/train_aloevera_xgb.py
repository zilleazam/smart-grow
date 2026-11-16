"""Train the XGBoost environmental-analysis model for aloe vera.

Inputs/targets: Nitrogen, Phosphorus, Potassium, Light (lux), Soil Moisture.
Note: as in the original script, the dataset and output file names still say "lettuce";
point DATASET_PATH / MODEL_OUT at your aloe vera files.
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATASET_PATH = "/content/lettuce_dataset_combined.xlsx"
MODEL_OUT = "lettuce_xgb_model.pkl"

# Load dataset
df = pd.read_excel(DATASET_PATH)

# Define features and dynamically find target variables
X = df[['Nitrogen', 'Phosphorus', 'Potassium', 'Light (lux)', 'Soil Moisture']]
corrected_target_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'Light (lux)', 'Soil Moisture']
y = df[corrected_target_columns]

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# Train with GridSearchCV
cv_model = GridSearchCV(xgb_model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
cv_model.fit(X_train, y_train)

# Save the best model
best_xgb_model = cv_model.best_estimator_
joblib.dump(best_xgb_model, MODEL_OUT)
print(f"\n✅ Model saved as '{MODEL_OUT}'")

# Evaluate model
y_pred = best_xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f" - Mean Absolute Error: {mae:.4f}")
print(f" - Mean Squared Error: {mse:.4f}")
print(f" - Root Mean Squared Error: {rmse:.4f}")
print(f" - R-squared Score: {r2:.4f}")

# Plot evaluation metrics
metrics = [mae, mse, rmse, r2]
labels = ["MAE", "MSE", "RMSE", "R-Squared"]
plt.figure(figsize=(8, 5))
plt.bar(labels, metrics, color=['blue', 'red', 'green', 'purple'])
plt.ylabel("Score")
plt.title("Model Evaluation Metrics")
plt.show()

# Prediction example
current_values = np.array([[10, 35, 99, 10.0, 2800]])  # Replace with real-time data
predicted_optimal = best_xgb_model.predict(current_values)

# Ensure predictions are in the correct shape
predicted_optimal = np.array(predicted_optimal).flatten()

# Calculate adjustments (difference between predicted optimal and current values)
adjustments = predicted_optimal - current_values[0]

# Print results
print("\n🌱 **Optimal Growth Conditions**")
print("\n🟢 **Current (Input) Values:**")
for feature, value in zip(X.columns, current_values[0]):
    print(f" - {feature}: {value}")

print("\n🟢 **Predicted Optimal Values:**")
for attr, value in zip(corrected_target_columns, predicted_optimal):
    print(f" - {attr}: {value:.2f}")

print("\n🔄 **Required Adjustments:**")
for attr, adj in zip(corrected_target_columns, adjustments):
    sign = "Increase" if adj > 0 else "Decrease"
    print(f" - {attr}: {sign} by {abs(adj):.2f}")

# Plot ideal vs adjusted values
x = np.arange(len(corrected_target_columns))  # x locations for the groups
width = 0.4  # Bar width

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, current_values[0][:len(corrected_target_columns)], width=width,
        alpha=0.6, label="Current Values", color='blue')
plt.bar(x + width / 2, predicted_optimal, width=width, alpha=0.6, label="Optimal Values", color='orange')

plt.xticks(x, corrected_target_columns)  # Set x-axis labels
plt.ylabel("Value")
plt.title("Current vs Optimal Values")
plt.legend()
plt.show()
