import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import requests

# Helsinki's latitude & longitude
LATITUDE = 60.1695
LONGITUDE = 24.9354

# Function to fetch historical weather
def fetch_historical_weather(start_date, end_date):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={LATITUDE}&longitude={LONGITUDE}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,cloud_cover_mean&timezone=Europe/Helsinki"
    response = requests.get(url)
    weather_data = response.json()["daily"]

    return pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date),
        "Temp_Max": weather_data["temperature_2m_max"],
        "Temp_Min": weather_data["temperature_2m_min"],
        "Precipitation": weather_data["precipitation_sum"],
        "Wind_Speed": weather_data["wind_speed_10m_max"],
        "Cloud_Cover": weather_data["cloud_cover_mean"]
    })

# Load sales data
data = pd.read_csv('uploads/data_without_weather.csv')

# Preprocess data
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['WeekOfYear'] = data['Date'].dt.isocalendar().week

# Fetch & merge historical weather
start_date = data["Date"].min().strftime("%Y-%m-%d")
end_date = data["Date"].max().strftime("%Y-%m-%d")
weather_data = fetch_historical_weather(start_date, end_date)
data = data.merge(weather_data, on="Date", how="left")

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['Drink', 'Event Type'])
data.fillna(data.mean(), inplace=True)  # Handle missing values

# Split data into training and testing sets
train_data = data.iloc[:-7]
test_data = data.iloc[-7:]

# Train the model
X_train = train_data.drop(columns=['Quantity Sold', 'Date'])
y_train = train_data['Quantity Sold']
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Test the model
X_test = test_data.drop(columns=['Quantity Sold', 'Date'])
y_test = test_data['Quantity Sold']
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Compare predicted vs true values
comparison = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred})
print(comparison)