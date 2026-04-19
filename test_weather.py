import requests
import pandas as pd

def test_weather_api():
    print("Fetching historical weather for Delhi...")
    
    # Delhi's coordinates
    lat, lon = 28.6139, 77.2090 
    
    # The Open-Meteo Historical API endpoint
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2024-01-01",
        "end_date": "2024-01-07",
        "daily": ["temperature_2m_max", "precipitation_sum", "wind_speed_10m_max"],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Convert the JSON response into a clean Pandas DataFrame
    weather_df = pd.DataFrame({
        "date": data["daily"]["time"],
        "max_temp_c": data["daily"]["temperature_2m_max"],
        "precipitation_mm": data["daily"]["precipitation_sum"],
        "max_wind_speed_kmh": data["daily"]["wind_speed_10m_max"]
    })

    print(weather_df)

if __name__ == "__main__":
    test_weather_api()