import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load dataset
df = pd.read_csv("synthetic_crop_dataset.csv")

# 2. Define features for crop recommendation
feature_cols = ["Region", "Year", "Temperature(°C)", "Rainfall(mm)", "Humidity(%)",
                "Soil_pH", "Nitrogen(N)", "Phosphorus(P)", "Potassium(K)", "Soil_Moisture(%)"]

X = df[feature_cols]
y = df['Crop']

# 3. Encode categorical variables
crop_encoder = LabelEncoder()
region_encoder = LabelEncoder()

y = crop_encoder.fit_transform(y)
X['Region'] = region_encoder.fit_transform(X['Region'])

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Save model, encoders, and feature columns
joblib.dump(model, "crop_model.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")
joblib.dump(region_encoder, "region_encoder.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")

print("✅ Model trained and saved successfully!")
