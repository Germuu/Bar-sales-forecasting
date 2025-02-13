from flask import Flask, render_template, request
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

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_filename, model, train_data, drinks, X_train_columns

    predictions = None  # Store predictions to pass to the template

    if request.method == "POST":
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

        elif "precipitation_0" in request.form:  # Prediction Form
            if model is None or X_train_columns is None:
                return "Please upload a dataset first.", 400

            today = datetime.today()

            # Find the next Wednesday from today
            days_until_next_wednesday = (2 - today.weekday()) % 7  # Wednesday = 2 (Monday=0)
            next_wednesday = today + timedelta(days=days_until_next_wednesday)

            # Generate future dates for prediction (Wednesday to Tuesday)
            future_dates = pd.date_range(next_wednesday, periods=7)

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

    return render_template("index.html", filename=uploaded_filename, drinks=drinks, predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
