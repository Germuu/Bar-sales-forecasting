import pandas as pd
from prophet import Prophet

def predict_sales(filepath):
    # Load and preprocess data
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Map numeric event types to categories
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
        
        # One-hot encode events
        prophet_df = pd.get_dummies(prophet_df, columns=['Event Type'], prefix='event')
        
        # Store regressor columns (excluding ds and y)
        regressor_cols = [col for col in prophet_df.columns if col not in ['ds', 'y']]
        
        # Initialize model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add regressors
        for col in regressor_cols:
            model.add_regressor(col)
        
        # Fit model
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=7, freq='D')
        
        # Add regressors (using placeholder values)
        future['Temperature'] = prophet_df['Temperature'].mean()
        future['Precipitation'] = prophet_df['Precipitation'].mean()
        future['Event Type'] = 'none'
        
        # Encode events
        future = pd.get_dummies(future, columns=['Event Type'], prefix='event')
        
        # Ensure all required columns exist
        missing_cols = set(regressor_cols) - set(future.columns)
        for col in missing_cols:
            future[col] = 0  # Add missing regressors with default value
            
        # Select only required columns
        future = future[['ds'] + regressor_cols]
        
        # Generate forecast
        forecast = model.predict(future)
        next_week_sales = forecast.tail(7)['yhat'].sum()
        
        predictions.append({
            'Drink': drink,
            'Predicted Sales': max(round(next_week_sales), 0),
            'Confidence': f"{round(forecast.tail(7)['yhat_lower'].mean())}-{round(forecast.tail(7)['yhat_upper'].mean())}"
        })

    return pd.DataFrame(predictions)