"""Train the Random Forest environmental-analysis model for lettuce.

Inputs: Nitrogen, Phosphorus, Potassium, Light (lux), Soil Moisture.
Targets: the adjustment needed for each input (difference between consecutive readings).
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset from Google Drive or local path
file_path = "/content/lettuce_dataset_combined.xlsx"  # Update path if needed
df = pd.read_excel(file_path)

# Convert Timestamp to datetime and sort
if 'Timestamp' in df.columns:
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')

# Define Features (X) and Target Adjustments (Y)
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Light (lux)', 'Soil Moisture']


def calculate_adjustments(df):
    """Calculates the required adjustment by checking the difference between consecutive timestamps."""
    adjustments = df[features].diff().fillna(0)  # First row has no previous data
    return adjustments


# Create target variables (adjustments needed)
df_adjustments = calculate_adjustments(df)

# Prepare data for training
X = df[features]  # Input features
y = df_adjustments  # Target adjustments

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using manual iteration with progress bar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

best_model = None
best_score = float('-inf')
best_params = None

print("Training model...")
param_list = list(ParameterGrid(param_grid))
with tqdm(total=len(param_list), desc="Grid Search Progress") as pbar:
    for params in param_list:
        rf = RandomForestRegressor(random_state=42, **params)
        rf.fit(X_train, y_train)
        score = rf.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = rf
            best_params = params
        pbar.update(1)

print(f"Best Parameters: {best_params}")

# Evaluate the model
predictions = best_model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared Score: {r2:.4f}")

# Save the trained model
joblib.dump(best_model, "lettuce_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved as lettuce_model.pkl and scaler.pkl")

# Load the trained model and scaler (for later use)
model = joblib.load("lettuce_model.pkl")
scaler = joblib.load("scaler.pkl")


def predict_adjustments(nitrogen, phosphorus, potassium, light, moisture):
    """Predicts the required adjustments based on current values and returns in dataset format."""
    current_values = [nitrogen, phosphorus, potassium, light, moisture]
    input_data = np.array([current_values])
    input_data_scaled = scaler.transform(input_data)
    predicted_adjustments = model.predict(input_data_scaled)[0]

    # Format output as a dictionary with integer values
    adjustments_dict = {features[i]: int(round(predicted_adjustments[i])) for i in range(len(features))}

    # Calculate ideal values by adding adjustments to the current values
    ideal_values = {features[i]: int(round(current_values[i] + predicted_adjustments[i]))
                    for i in range(len(features))}

    return adjustments_dict, ideal_values


# Example Prediction
current_values = [110, 305, 299, 10.0, 2800]  # Replace with real-time data
adjustments, ideal_values = predict_adjustments(*current_values)
print("Predicted Adjustments:")
print(adjustments)
print("Ideal Values:")
print(ideal_values)

# Plot accuracy graph
plt.figure(figsize=(10, 6))
sns.barplot(x=['MAE', 'MSE', 'RMSE', 'R2'], y=[mae, mse, rmse, r2])
plt.title("Model Accuracy Metrics")
plt.ylabel("Error Value")
plt.show()
