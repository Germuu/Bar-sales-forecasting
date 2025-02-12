import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from src.models.predictor import predict_sales  # Import your existing predictor function

# Linear sales function to simulate sales
def linear_sales_function(temperature, precipitation, event_type, drink_type, base_sales=50):
    """
    A linear function to simulate sales based on temperature, precipitation, event type, and drink type.
    """
    event_factor = {'none': 1, 'music': 1.2, 'other': 1.1}  # Event impact on sales (shared across all drinks)
    
    # Drink-specific effects
    drink_factors = {
        'Vodka': {'temperature': 0.6, 'precipitation': 0.3, 'event_factor': {'none': 1, 'music': 1.8, 'other': 1.2}},
        'Whiskey': {'temperature': 0.8, 'precipitation': 0.2, 'event_factor': {'none': 1, 'music': 1.3, 'other': 0.9}},
        'Beer': {'temperature': 1.0, 'precipitation': -0.5, 'event_factor': {'none': 1, 'music': 1.5, 'other': 1.0}},
    }
    
    # Adjust the factors based on the drink type
    drink_specific_factors = drink_factors.get(drink_type, {'temperature': 0.5, 'precipitation': 0.3, 'event_factor': event_factor})
    
    # Impact of temperature and precipitation on sales for this drink
    temperature_effect = drink_specific_factors['temperature'] * temperature
    precipitation_effect = drink_specific_factors['precipitation'] * precipitation
    
    # Event-specific effect on sales (this depends on both the event and the drink)
    event_effect = drink_specific_factors['event_factor'].get(event_type, 1) * 10  # 10 is a scaling factor for event impact
    
    # Calculate sales using a linear combination
    sales = base_sales + temperature_effect + precipitation_effect + event_effect
    
    return sales

# Generate synthetic data for 3 months with consecutive days, with interdependencies
def generate_synthetic_data(start_date="2024-01-01", days=90):
    start_date = pd.to_datetime(start_date)
    dates = pd.date_range(start_date, periods=days, freq='D')
    
    drinks = ['Vodka', 'Whiskey', 'Beer']
    event_types = ['none', 'music', 'other']
    data = []

    for date in dates:
        for drink in drinks:
            # Random values for temperature, precipitation, and event type
            temperature = np.random.uniform(10, 30)  # Random temperature between 10 and 30°C
            precipitation = np.random.uniform(0, 20)  # Random precipitation between 0 and 20 mm
            event_type = np.random.choice(event_types)

            # Apply the linear sales function with interdependencies
            sales = linear_sales_function(temperature, precipitation, event_type, drink)

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

# Print first few rows to check data
print(synthetic_data.head())


# Split the data into training and testing sets (first 83 days for training, last 7 days for testing)
train_data = synthetic_data[synthetic_data['Date'] < '2024-03-22']  # First 83 days
test_data = synthetic_data[synthetic_data['Date'] >= '2024-03-22']  # Last 7 days

# Print sizes of training and test sets
print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")

# Save the train_data to a temporary CSV file (optional for the predictor function)
train_file_path = 'train_data.csv'
train_data.to_csv(train_file_path, index=False)

# Use the existing predict_sales function to predict sales
predictions = predict_sales(train_file_path)

# Calculate the Mean Squared Error (MSE) on the test data
mse_loss = 0
for drink in predictions['Drink']:
    # Get the predicted sales
    predicted_sales = predictions[predictions['Drink'] == drink]['Predicted Sales'].values[0]
    
    # Get the true sales for the corresponding drink in the test data (last 7 days)
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
    true_sales_avg = test_data[test_data['Drink'] == drink]['Quantity Sold'].tail(7).sum()
    
    print(f'{drink}: Predicted Sales = {predicted_sales:.2f}, True Sales = {true_sales_avg:.2f}')
