import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

# Step 1: Load the dataset
df = pd.read_csv('Datasets/house-price-prediction-dataset/House Price Prediction Dataset.csv')

# Step 2: Encode categorical fields
label_cols = ['Garage', 'Condition', 'Location']

# Create label encoders for each field
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 3: Select features and target
features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Garage', 'Location', 'Condition']
target = 'Price'

X = df[features]
y = df[target]

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
# print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 7: Save the trained model
dump(model, 'house_price_model_final.joblib')

print("Model trained and saved successfully!")



