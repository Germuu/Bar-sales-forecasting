import pandas as pd
from prophet import Prophet

def preprocess_data(data):
    # Ensure the Date column is in datetime format
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # Fill missing values in Temperature and Precipitation with the median
    data['Temperature'] = data['Temperature'].fillna(data['Temperature'].median())
    data['Precipitation'] = data['Precipitation'].fillna(data['Precipitation'].median())
    
    # Convert Event Type column to categorical labels
    event_mapping = {'none': 0, 'music': 1, 'other': 2}
    data['Event Type'] = data['Event Type'].map(event_mapping)
    
    # Ensure Quantity Sold is numeric
    data['Quantity Sold'] = pd.to_numeric(data['Quantity Sold'], errors='coerce')
    
    # Drop rows with missing values in critical columns
    data.dropna(subset=['Quantity Sold', 'Date'], inplace=True)
    
    return data

def predict_sales(filepath, event_data=None):
    # Load and preprocess data
    data = pd.read_csv(filepath)
    data = preprocess_data(data)  # Apply preprocessing to ensure correct format
    
    # Map numeric event types back to their category names for future use
    event_mapping = {0: 'none', 1: 'music', 2: 'other'}
    data['Event Type'] = data['Event Type'].map(event_mapping)
    
    predictions = []
    
    for drink in data['Drink'].unique():
        drink_data = data[data['Drink'] == drink].copy()
        
        # Prepare Prophet dataframe
        prophet_df = drink_data[['Date', 'Quantity Sold', 'Temperature', 
                                 'Precipitation', 'Event Type']].rename(columns={
                                     'Date': 'ds',
                                     'Quantity Sold': 'y'
                                 })
        
        # One-hot encode event type in training data
        prophet_df = pd.get_dummies(prophet_df, columns=['Event Type'], prefix='event')

        # Store regressor columns (excluding ds and y)
        regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add regressors (temperature, precipitation, and all event type columns)
        for col in regressor_cols:
            model.add_regressor(col)

        # Fit the model
        model.fit(prophet_df)

        # Create future dataframe (next week prediction)
        future = model.make_future_dataframe(periods=7, freq='D')

        # Add regressors with default values
        future['Temperature'] = prophet_df['Temperature'].mean()
        future['Precipitation'] = prophet_df['Precipitation'].mean()

        # **Corrected: Ensure event columns exist in the future dataframe**
        future_events = pd.DataFrame(0, index=future.index, columns=[col for col in regressor_cols if 'event_' in col])

        # Assign provided event data (only for last 7 days)
        if event_data:
            event_mapping = {'none': [1, 0, 0], 'music': [0, 1, 0], 'other': [0, 0, 1]}
            event_encoded = [event_mapping.get(evt, [0, 0, 0]) for evt in event_data]

            # Convert list to DataFrame
            future_events.iloc[-7:] = event_encoded

        # Merge event regressors into future dataframe
        future = pd.concat([future, future_events], axis=1)


        # Generate forecast
        forecast = model.predict(future)
        
        # Sum predicted sales for the next week
        next_week_sales = forecast.tail(7)['yhat'].sum()
        
        predictions.append({
            'Drink': drink,
            'Predicted Sales': max(round(next_week_sales), 0),
            'Confidence': f"{round(forecast.tail(7)['yhat_lower'].mean())}-{round(forecast.tail(7)['yhat_upper'].mean())}"
        })

    return pd.DataFrame(predictions)
