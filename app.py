from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import numpy as np
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

@app.route("/", methods=["GET"])
def index():
    global drinks, uploaded_filename
    return render_template("index.html", filename=uploaded_filename, drinks=drinks)

@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_filename, model, train_data, drinks, X_train_columns

    if "file" in request.files:
        file = request.files["file"]
        if file.filename:
            uploaded_filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(uploaded_filename)

            # Load and preprocess dataset
            train_data = pd.read_csv(uploaded_filename)
            train_data['Date'] = pd.to_datetime(train_data['Date'])
            train_data['DayOfWeek'] = train_data['Date'].dt.dayofweek
            train_data['Month'] = train_data['Date'].dt.month
            train_data['WeekOfYear'] = train_data['Date'].dt.isocalendar().week

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

    return redirect(url_for('setup_prediction'))  # Redirect to date selection

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

        return redirect(url_for('weather_input'))

    return render_template("setup_prediction.html")

@app.route("/weather_input", methods=["GET"])
def weather_input():
    if "future_dates" not in session:
        return redirect(url_for("setup_prediction"))  # Redirect if dates are missing

    future_dates = [datetime.strptime(date, '%Y-%m-%d') for date in session['future_dates']]
    return render_template("weather_input.html", future_dates=future_dates)

@app.route("/predict", methods=["POST"])
def predict():
    global model, X_train_columns, drinks

    if model is None or X_train_columns is None:
        return "Please upload a dataset first.", 400

    if "future_dates" not in session:
        return redirect(url_for("setup_prediction"))

    future_dates = [datetime.strptime(date, '%Y-%m-%d') for date in session['future_dates']]
    
    # Create DataFrame for predictions
    next_week_data = []
    for i, future_date in enumerate(future_dates):
        for drink in drinks:
            next_week_data.append({
                "DayOfWeek": future_date.weekday(),
                "Month": future_date.month,
                "WeekOfYear": future_date.isocalendar().week,
                "Precipitation": float(request.form[f"precipitation_{i}"]),
                "Temperature": float(request.form[f"temperature_{i}"]),
                **{f"Drink_{drink}": 1 if d == drink else 0 for d in drinks},
                "Event Type_music": 1 if request.form[f"event_{i}"] == "music" else 0,
                "Event Type_none": 1 if request.form[f"event_{i}"] == "none" else 0,
                "Event Type_other": 1 if request.form[f"event_{i}"] == "other" else 0,
            })

    next_week_df = pd.DataFrame(next_week_data)
    next_week_df = next_week_df[X_train_columns]  # Ensure correct feature order
    predictions_raw = model.predict(next_week_df)

    # Store predictions per drink for each day
    predictions = []
    total_sales = {drink: 0 for drink in drinks}  # Initialize total sales

    for i, future_date in enumerate(future_dates):
        daily_predictions = {
            "date": future_date.strftime("%A, %d %B %Y"),
            "sales": {drink: round(predictions_raw[i * len(drinks) + j]) for j, drink in enumerate(drinks)}
        }
        predictions.append(daily_predictions)

        # Sum up the total sales per drink
        for drink, sales in daily_predictions["sales"].items():
            total_sales[drink] += sales

    return render_template("predictions.html", predictions=predictions, total_sales=total_sales)

