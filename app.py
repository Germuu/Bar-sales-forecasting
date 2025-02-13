from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

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
    global drinks, uploaded_filename, model, train_data

    # Just render the home page
    return render_template("index.html", filename=uploaded_filename, drinks=drinks, predictions=None)

@app.route("/upload", methods=["POST"])
def upload():
    global uploaded_filename, model, train_data, drinks, X_train_columns

    if "file" in request.files:  # File Upload Form
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

    # After uploading, redirect to the index page
    return redirect(url_for('index'))

@app.route("/predict", methods=["POST"])
def predict():
    global model, X_train_columns, drinks

    if model is None or X_train_columns is None:
        return "Please upload a dataset first.", 400

    # Get user input for the start date and number of days
    start_date_str = request.form["start_date"]
    num_days = int(request.form["num_days"])

    # Convert start date string to a datetime object
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")

    # Generate the future dates based on the user input
    future_dates = pd.date_range(start_date, periods=num_days)

    # Create the DataFrame with correct features
    next_week_data = []
    for i, future_date in enumerate(future_dates):
        for drink in drinks:
            next_week_data.append({
                "DayOfWeek": future_date.dayofweek,
                "Month": future_date.month,
                "WeekOfYear": future_date.isocalendar().week,
                "Precipitation": float(request.form[f"precipitation_{i}"]),
                "Temperature": float(request.form[f"temperature_{i}"]),
                **{f"Drink_{drink}": 1 if d == drink else 0 for d in drinks},
                "Event Type_music": 1 if request.form[f"event_{i}"] == "music" else 0,
                "Event Type_none": 1 if request.form[f"event_{i}"] == "none" else 0,
                "Event Type_other": 1 if request.form[f"event_{i}"] == "other" else 0,
            })

    # Convert to DataFrame
    next_week_df = pd.DataFrame(next_week_data)

    # Ensure feature order matches training data
    next_week_df = next_week_df[X_train_columns]

    # Make predictions
    predictions_raw = model.predict(next_week_df)

    # Aggregate predictions per drink
    predictions = {drink: round(sum(predictions_raw[i::len(drinks)])) for i, drink in enumerate(drinks)}

    # Render the page with predictions
    return render_template("index.html", filename=uploaded_filename, drinks=drinks, predictions=predictions)


if __name__ == "__main__":
    app.run(debug=True)
