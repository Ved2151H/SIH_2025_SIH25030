import joblib
import pandas as pd

# Load trained model, encoders, and feature columns
model = joblib.load("crop_model.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
region_encoder = joblib.load("region_encoder.pkl")
feature_cols = joblib.load("feature_columns.pkl")

# Function to safely encode region
def encode_region_safe(region):
    if region in region_encoder.classes_:
        return region_encoder.transform([region])[0]
    else:
        print(f"⚠️ Warning: '{region}' not in training regions. Using default region '{region_encoder.classes_[0]}'")
        return 0  # default to first region in training

# Function for crop recommendation
def recommend_crop(region, year, temperature, rainfall, humidity, soil_ph, nitrogen, phosphorus, potassium, moisture):
    region_encoded = encode_region_safe(region)

    # Create input dataframe
    input_data = pd.DataFrame([[
        region_encoded, year, temperature, rainfall, humidity, soil_ph,
        nitrogen, phosphorus, potassium, moisture
    ]], columns=feature_cols)

    # Predict
    prediction = model.predict(input_data)[0]
    crop_name = crop_encoder.inverse_transform([prediction])[0]
    return crop_name

# CLI loop for user input
if __name__ == "__main__":
    while True:
        print("\n🌱 Enter field data to get crop recommendation (or type 'exit' to quit):")

        region = input("Region: ")
        if region.lower() == "exit":
            break
        year = int(input("Year: "))
        temperature = float(input("Temperature (°C): "))
        rainfall = float(input("Rainfall (mm): "))
        humidity = float(input("Humidity (%): "))
        soil_ph = float(input("Soil pH: "))
        nitrogen = float(input("Nitrogen (N): "))
        phosphorus = float(input("Phosphorus (P): "))
        potassium = float(input("Potassium (K): "))
        moisture = float(input("Soil Moisture (%): "))

        crop = recommend_crop(region, year, temperature, rainfall, humidity, soil_ph, nitrogen, phosphorus, potassium, moisture)
        print(f"✅ Recommended Crop: {crop}")
