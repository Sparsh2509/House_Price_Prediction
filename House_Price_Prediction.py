import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

# Step 1: Load the dataset
df = pd.read_csv(r'D:\Sparsh\ML_Projects\House_Price_Prediction\Datasets\house-price-prediction-dataset\House Price Prediction dataset.csv')

# Step 2: Convert categorical columns to lowercase
categorical_cols = ['Garage', 'Condition', 'Location']
for col in categorical_cols:
    df[col] = df[col].str.lower()

# Step 3: Encode categorical fields
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Select features and target
features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Garage', 'Location', 'Condition']
target = 'Price'

X = df[features]
y = df[target]

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 8: Save the trained model and label encoders
dump(model, 'house_price_model_final.pkl')
dump(label_encoders, 'label_encoders.pkl')

print("Model and encoders saved successfully!")
