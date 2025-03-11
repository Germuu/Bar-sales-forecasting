import requests
from datetime import datetime

# Helsinki's latitude & longitude (you can change this to your location)
LATITUDE = 60.1695
LONGITUDE = 24.9354

# Function to fetch the weather forecast for a specific date
def fetch_weather_forecast(date):
    # Ensure the date is in the correct format (YYYY-MM-DD)
    if isinstance(date, datetime):
        date = date.strftime('%Y-%m-%d')
    
    # API URL to fetch the weather forecast for a given date
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&daily=temperature_2m_max,temperature_2m_min&timezone=Europe/Helsinki&start_date={date}&end_date={date}"
    
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching weather data: {response.status_code}")
        return None

    weather_data = response.json().get("daily", {})
    
    # Check if the weather data contains the required information
    if not weather_data or date not in weather_data["time"]:
        print("Weather data for the given date is not available.")
        return None
    
    max_temp = weather_data["temperature_2m_max"][0]  # Max temperature for the date
    min_temp = weather_data["temperature_2m_min"][0]  # Min temperature for the date

    return max_temp, min_temp


if __name__ == "__main__":
    # Example usage
    input_date = datetime(2025, 2, 15)  # Example: Get weather for February 15, 2025
    weather = fetch_weather_forecast(input_date)
    
    if weather:
        max_temp, min_temp = weather
        print(f"Max Temperature: {max_temp}°C, Min Temperature: {min_temp}°C")
    else:
        print("Failed to retrieve weather data.")
