import pandas as pd
from prophet import Prophet

def predict_sales(filepath):
    # Load and preprocess data
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    
    predictions = []
    
    for drink in data['Drink'].unique():
        drink_data = data[data['Drink'] == drink].copy()
        
        # Prepare Prophet dataframe with only Date and Quantity Sold
        prophet_df = drink_data[['Date', 'Quantity Sold']].rename(columns={
            'Date': 'ds',
            'Quantity Sold': 'y'
        })
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Fit model
        model.fit(prophet_df)
        
        # Create future dataframe (predict for the next 7 days)
        future = model.make_future_dataframe(periods=7, freq='D')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Sum predicted sales for the next week
        next_week_sales = forecast.tail(7)['yhat'].sum()
        
        # Collect the prediction and confidence interval
        predictions.append({
            'Drink': drink,
            'Predicted Sales': max(round(next_week_sales), 0),  # Ensure non-negative predictions
            'Confidence': f"{round(forecast.tail(7)['yhat_lower'].mean())}-{round(forecast.tail(7)['yhat_upper'].mean())}"
        })

    return pd.DataFrame(predictions)
