import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Read the dataset
data = pd.read_csv('train_data.csv')

# Step 2: Handle date-related features
data['Date'] = pd.to_datetime(data['Date'])
data['DayOfWeek'] = data['Date'].dt.dayofweek
data['Month'] = data['Date'].dt.month
data['WeekOfYear'] = data['Date'].dt.isocalendar().week

# Step 3: Handle categorical features (e.g., using one-hot encoding for 'Drink' and 'Event Type')
data = pd.get_dummies(data, columns=['Drink', 'Event Type'])

# Step 4: Handle missing values (e.g., filling NaN with the mean for numeric columns)
data.fillna(data.mean(), inplace=True)

# Step 5: Create rolling averages for the past 7 days (assuming temperature and precipitation are numeric)


# Step 6: Prepare data for training and testing
# Get the unique dates in the dataset
unique_dates = data['Date'].unique()
# Select the last 7 unique dates for the test set
last_7_dates = unique_dates[-7:]  

# Filter data for the last 7 unique dates (keep all drinks for those days)
train_data = data[~data['Date'].isin(last_7_dates)]  # All except last 7 unique dates
test_data = data[data['Date'].isin(last_7_dates)]  # Data for the last 7 unique dates

# Prepare training data
X_train = train_data.drop(columns=['Quantity Sold', 'Date'])  # Features for training
y_train = train_data['Quantity Sold']  # Target (Sales) for training

# Prepare testing data (actual sales for the last 7 unique dates)
X_test = test_data.drop(columns=['Quantity Sold', 'Date'])  # Features for testing
y_test = test_data['Quantity Sold']  # Actual sales for testing

# Step 7: Train the model (Random Forest)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Step 8: Make predictions for all days (including the last 7 unique days) based on the trained model
y_pred_all_days = model.predict(data.drop(columns=['Quantity Sold', 'Date']))  # Predict for all days

# Step 9: Evaluate the model for the last 7 unique days
y_pred_last_7_days = y_pred_all_days[-len(last_7_dates)*3:]  # Get predictions for last 7 unique days (3 drinks per day)
mse = mean_squared_error(y_test, y_pred_last_7_days)
print(f'Mean Squared Error (MSE) for last 7 unique days: {mse}')

# Step 10: Optionally, print predicted vs actual sales for the last 7 unique days
predicted_sales = pd.DataFrame({
    'Date': test_data['Date'],
    'Predicted Sales': y_pred_last_7_days,
    'Actual Sales': y_test
})
print(predicted_sales)



# Plot feature importance for the Random Forest model
feature_importances = model.feature_importances_
features = X_train.columns
plt.barh(features, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()
