import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv(r'D:\Sparsh\ML_Projects\House_Price_Prediction\Datasets\house-price-prediction-dataset\House Price Prediction dataset.csv')

# Categorical columns to encode
label_cols = ['Garage', 'Condition', 'Location']
label_encoders = {}

# Fit LabelEncoders
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
features = ['Area','Bedrooms','Bathrooms','Floors','YearBuilt','Garage','Location','Condition']
target = 'Price'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Save model and encoders
dump(model, 'house_price_model_final.pkl')
dump(label_encoders, 'label_encoders.pkl')

print("Model and encoders saved successfully!")
