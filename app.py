from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = "supersecretkey"  # Required for session storage

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
uploaded_filename = None
model = None
train_data = None
drinks = []  # Stores unique drink names
X_train_columns = None  # Store feature columns from training

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


# Route: Home
@app.route("/", methods=["GET"])
def index():
    global drinks, uploaded_filename
    return render_template("index.html", filename=uploaded_filename, drinks=drinks)


# Route: Upload CSV
@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_filename, model, train_data, drinks, X_train_columns

    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            uploaded_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(uploaded_filename)

            # Load sales data
            train_data = pd.read_csv(uploaded_filename)

            train_data['Date'] = pd.to_datetime(train_data['Date'])
            train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
            train_data['Month'] = train_data['Date'].dt.month
            train_data['WeekOfYear'] = train_data['Date'].dt.isocalendar().week

            # Fetch & merge historical weather
            start_date = train_data["Date"].min().strftime("%Y-%m-%d")
            end_date = train_data["Date"].max().strftime("%Y-%m-%d")
            weather_data = fetch_historical_weather(start_date, end_date)
            train_data = train_data.merge(weather_data, on="Date", how="left")

            # One-hot encode categorical features
            train_data = pd.get_dummies(train_data, columns=['Drink', 'Event Type'])
            train_data.fillna(train_data.mean(), inplace=True)  # Handle missing values

            # Train the model
            X_train = train_data.drop(columns=['Quantity Sold', 'Date'])
            y_train = train_data['Quantity Sold']
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            # Store column order
            X_train_columns = X_train.columns

            # Extract unique drink names
            drinks = list(pd.read_csv(uploaded_filename)['Drink'].unique())

    return redirect(url_for('setup_prediction'))


# Route: Setup Prediction
@app.route("/setup_prediction", methods=["GET", "POST"])
def setup_prediction():
    if request.method == "POST":
        start_date = request.form.get("start_date")
        num_days = int(request.form.get("num_days"))

        if not start_date:
            return "Start date is required.", 400

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        future_dates = [start_date + timedelta(days=i) for i in range(num_days)]

        session['future_dates'] = [date.strftime('%Y-%m-%d') for date in future_dates]  # Store dates in session

        # Render the event input page with future_dates passed in context
        return render_template('weather_input.html', future_dates=future_dates)

    return render_template("setup_prediction.html")



# Function to fetch weather forecasts for future dates
def fetch_forecast_weather(start_date, end_date):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,cloud_cover_mean&timezone=Europe/Helsinki&start_date={start_date}&end_date={end_date}"
    response = requests.get(url)
    weather_forecast = response.json()["daily"]

    return pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date),
        "Temp_Max": weather_forecast["temperature_2m_max"],
        "Temp_Min": weather_forecast["temperature_2m_min"],
        "Precipitation": weather_forecast["precipitation_sum"],
        "Wind_Speed": weather_forecast["wind_speed_10m_max"],
        "Cloud_Cover": weather_forecast["cloud_cover_mean"]
    })


# Route: Predict Sales
@app.route("/predict", methods=["GET", "POST"])
def predict():
    global model, X_train_columns, drinks

    if model is None or X_train_columns is None:
        return "Please upload a dataset first.", 400

    if "future_dates" not in session:
        return redirect(url_for("setup_prediction"))

    future_dates = [datetime.strptime(date, "%Y-%m-%d") for date in session['future_dates']]

    if request.method == "POST":
        events_data = {}
        for i in range(len(future_dates)):
            event = request.form.get(f"event_{i}")
            events_data[future_dates[i]] = event  # Store event for the specific date

        # Fetch weather forecast
        start_date = future_dates[0].strftime("%Y-%m-%d")
        end_date = future_dates[-1].strftime("%Y-%m-%d")
        weather_forecast = fetch_forecast_weather(start_date, end_date)

        # Prepare DataFrame for predictions
        next_week_data = []
        for i, future_date in enumerate(future_dates):
            event = events_data[future_date]  # Get the event for the specific date
            
            for drink in drinks:
                next_week_data.append({
                    "DayOfWeek": future_date.weekday(),
                    "Month": future_date.month,
                    "WeekOfYear": future_date.isocalendar().week,
                    "Temp_Max": weather_forecast.loc[i, "Temp_Max"],
                    "Temp_Min": weather_forecast.loc[i, "Temp_Min"],
                    "Precipitation": weather_forecast.loc[i, "Precipitation"],
                    "Wind_Speed": weather_forecast.loc[i, "Wind_Speed"],
                    "Cloud_Cover": weather_forecast.loc[i, "Cloud_Cover"],
                    **{f"Drink_{drink}": 1 if d == drink else 0 for d in drinks},
                    # Set event types based on the selected event
                    "Event Type_ music": 1 if event == "music" else 0,
                    "Event Type_ none": 1 if event == "none" else 0,
                    "Event Type_ other": 1 if event == "other" else 0,
                })

        next_week_df = pd.DataFrame(next_week_data)
        next_week_df = next_week_df[X_train_columns]  # Ensure correct feature order
        predictions_raw = model.predict(next_week_df)

        # Store predictions per drink for each day
        predictions = []
        total_sales = {drink: 0 for drink in drinks}

        for i, future_date in enumerate(future_dates):
            daily_predictions = {
                "date": future_date.strftime("%A, %d %B %Y"),
                "sales": {drink: round(predictions_raw[i * len(drinks) + j]) for j, drink in enumerate(drinks)}
            }
            predictions.append(daily_predictions)

            for drink, sales in daily_predictions["sales"].items():
                total_sales[drink] += sales

        return render_template("predictions.html", predictions=predictions, total_sales=total_sales)



if __name__ == "__main__":
    app.run(debug=True)
