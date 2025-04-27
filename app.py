
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as mae

# Load your dataset
@st.cache_data
def load_data():
    df = pd.read_csv('retail_store_inventory.csv')  # Make sure this file exists
    return df

# Load data
df = load_data()

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

df['Discounted Price'] = df['Price'] * (1 - df['Discount'] / 100)
df['Price Difference'] = df['Price'] - df['Competitor Pricing']
df['Stock to Order Ratio'] = df['Inventory Level'] / (df['Units Ordered'] + 1)
df['Forecast Accuracy'] = abs(df['Demand Forecast'] - df['Units Sold']) / (df['Units Sold'] + 1)

def classify_units(units):
    if units <= 50:
        return 0  # Low
    elif units <= 150:
        return 1  # Medium
    else:
        return 2  # High

df['Demand Class'] = df['Units Sold'].apply(classify_units)

# Prepare features and target
feature_cols = [
    'Price', 'Discount', 'Demand Forecast', 'Competitor Pricing',
    'Discounted Price', 'Price Difference', 'Stock to Order Ratio',
    'Forecast Accuracy', 'Holiday/Promotion', 'Year', 'Month', 'Day'
]

X = df[feature_cols]
y = df['Demand Class']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=22)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Models
models = [
    ("Linear Regression", LinearRegression()),
    ("Lasso Regression", Lasso(alpha=0.1)),
    ("Ridge Regression", Ridge(alpha=1.0))
]

trained_models = {}
for name, model in models:
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

# Streamlit UI
st.title("Retail Store Demand Prediction")

# Create input fields
price = st.number_input("Price", min_value=0.0, value=100.0)
discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, value=10.0)
demand_forecast = st.number_input("Demand Forecast (Units)", min_value=0.0, value=100.0)
competitor_pricing = st.number_input("Competitor Pricing ($)", min_value=0.0, value=90.0)
inventory_level = st.number_input("Inventory Level", min_value=0.0, value=500.0)
units_ordered = st.number_input("Units Ordered", min_value=0, value=50)

# Predict button
if st.button("Predict Demand Class"):
    input_data = pd.DataFrame({
        'Price': [price],
        'Discount': [discount],
        'Demand Forecast': [demand_forecast],
        'Competitor Pricing': [competitor_pricing],
        'Discounted Price': [price * (1 - discount / 100)],
        'Price Difference': [price - competitor_pricing],
        'Stock to Order Ratio': [inventory_level / (units_ordered + 1)],
        'Forecast Accuracy': [abs(demand_forecast - units_ordered) / (units_ordered + 1)],
        'Holiday/Promotion': [0],  # Default value
        'Year': [2025],             # Default value
        'Month': [5],               # Default value
        'Day': [15]                 # Default value
    })

    input_scaled = scaler.transform(input_data)

    predictions = {}
    for name, model in trained_models.items():
        pred = model.predict(input_scaled)
        predictions[name] = pred[0]

    st.subheader("Predicted Demand Classes")
    for name, pred in predictions.items():
        st.write(f"{name}: {pred:.2f}")

    st.subheader("Model Performance (Training and Validation Errors)")
    for name, model in trained_models.items():
        train_preds = model.predict(X_train_scaled)
        train_error = mae(y_train, train_preds)

        val_preds = model.predict(X_val_scaled)
        val_error = mae(y_val, val_preds)

        st.write(f"{name}:")
        st.write(f"  Training Error (MAE): {train_error:.4f}")
        st.write(f"  Validation Error (MAE): {val_error:.4f}")
