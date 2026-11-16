"""Predict ideal growth values and required adjustments from a CSV of sensor readings.

The thesis used the same script for lettuce and aloe vera, changing only the model file.

Usage: python predict_from_csv.py <model.pkl> <sensor_data.csv>
CSV columns: Soil Moisture, Light Intensity, N, P, K
"""
import sys
import pandas as pd
import joblib

model_path = sys.argv[1] if len(sys.argv) > 1 else "models/sensor_lettuce_trained_model.pkl"
file_path = sys.argv[2] if len(sys.argv) > 2 else "sensor_data.csv"

# Load the trained model
model = joblib.load(model_path)

# Load the new values from CSV
df = pd.read_csv(file_path)

# Extract relevant features for prediction
features = ["Soil Moisture", "Light Intensity", "N", "P", "K"]
current_values = df[features].values  # Convert to NumPy array

# Predict adjustments (difference from ideal values)
predicted_adjustments = model.predict(current_values)

# Compute ideal values (Ideal = Current + Adjustments)
predicted_ideal_values = current_values + predicted_adjustments

# Display results in the required format
for i in range(len(df)):
    print("\n🌱 **Optimal Growth Recommendations**")

    print("\n🟢 **Current Values:**")
    for feature, value in zip(features, current_values[i]):
        print(f" - {feature}: {value:.2f}")

    print("\n🌟 **Predicted Ideal Values (Optimal Growth Levels):**")
    for feature, ideal_value in zip(features, predicted_ideal_values[i]):
        print(f" - {feature}: {ideal_value:.2f}")

    print("\n🔄 **Required Adjustments (Predicted by Model):**")
    for feature, adj in zip(features, predicted_adjustments[i]):
        sign = "Increase" if adj > 0 else "Decrease"
        print(f" - {feature}: {sign} by {abs(adj):.2f}")

    print("\n" + "-" * 50)  # Separator for better readability
