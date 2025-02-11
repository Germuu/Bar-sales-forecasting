import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.models.predictor import predict_sales  # Import your existing predictor function

# Linear sales function to simulate sales
def linear_sales_function(temperature, precipitation, event_type, base_sales=50):
    """
    A linear function to simulate sales based on temperature, precipitation, and event type.
    """
    event_factor = {'none': 1, 'music': 1.2, 'other': 1.1}  # Linear impact of event type
    temperature_factor = 0.5  # Impact of temperature on sales (positive correlation)
    precipitation_factor = -0.3  # Impact of precipitation on sales (negative correlation)

    # Linear combination of features
    sales = base_sales + (temperature_factor * temperature) + (precipitation_factor * precipitation) + (event_factor[event_type] * 10)
    
    return sales

# Generate synthetic data for 3 months with consecutive days
def generate_synthetic_data(start_date="2024-01-01", days=90):
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start_date, periods=days, freq='D')
    
    drinks = ['Beer', 'Vodka', 'Whiskey']
    event_types = ['none', 'music', 'other']
    data = []

    for date in dates:
        for drink in drinks:
            # Random values for temperature, precipitation, and event type
            temperature = np.random.uniform(10, 30)  # Random temperature between 10 and 30°C
            precipitation = np.random.uniform(0, 20)  # Random precipitation between 0 and 20 mm
            event_type = np.random.choice(event_types)

            # Apply the linear sales function
            sales = linear_sales_function(temperature, precipitation, event_type)

            data.append({
                'Date': date,
                'Drink': drink,
                'Temperature': temperature,
                'Precipitation': precipitation,
                'Event Type': event_type,
                'Quantity Sold': sales
            })

    return pd.DataFrame(data)

# Generate synthetic data for 3 months
synthetic_data = generate_synthetic_data(start_date="2024-01-01", days=90)

# Sort the data by Date to ensure Prophet gets data in the correct order
synthetic_data = synthetic_data.sort_values(by='Date')

# Split the data into training and test sets (80% training, 20% testing)
train_data, test_data = train_test_split(synthetic_data, test_size=0.2, random_state=42)

# Save the train_data to a temporary CSV file (optional for the predictor function)
train_file_path = 'train_data.csv'
train_data.to_csv(train_file_path, index=False)

# Use the existing predict_sales function to predict sales
predictions = predict_sales(train_file_path)

# Calculate the Mean Squared Error (MSE) on the test data
# We will compare the predicted sales from the model with the actual sales in the test data

mse_loss = 0
for drink in predictions['Drink']:
    # Get the predicted sales
    predicted_sales = predictions[predictions['Drink'] == drink]['Predicted Sales'].values[0]
    
    # Get the true sales for the corresponding drink in the test data
    true_sales = test_data[test_data['Drink'] == drink]['Quantity Sold'].tail(7).sum()  # True sales over the last 7 days
    
    # Compute MSE (Mean Squared Error)
    mse_loss += (predicted_sales - true_sales) ** 2

# Calculate average MSE loss across all drinks
mse_loss /= len(predictions)

# Print out the results
print(f'Mean Squared Error (MSE) Loss: {mse_loss:.2f}')

# Display some true vs predicted examples
for drink in predictions['Drink']:
    predicted_sales = predictions[predictions['Drink'] == drink]['Predicted Sales'].values[0]
    true_sales_avg = test_data[test_data['Drink'] == drink]['Quantity Sold'].tail(7).mean()
    
    print(f'{drink}: Predicted Sales = {predicted_sales:.2f}, True Sales (avg) = {true_sales_avg:.2f}')
