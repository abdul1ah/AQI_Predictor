import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import GLOBAL_CITIES, HISTORICAL_YEARS

# 1. Set up a resilient session that automatically retries failed/timed-out requests
session = requests.Session()
retries = Retry(
    total=5,  # Try 5 times before giving up
    backoff_factor=1,  # Wait 1s, 2s, 4s, 8s between retries to give the API a break
    status_forcelist=[429, 500, 502, 503, 504], # Retry on rate limits and server errors
)
session.mount('https://', HTTPAdapter(max_retries=retries))

def get_coordinates(city_name: str):
    """Resolves latitude and longitude for a given city name."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    
    # 2. Use the session with a strict 10-second timeout
    response = session.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if not data.get("results"):
        raise ValueError(f"Could not resolve coordinates for {city_name}")
        
    return data["results"][0]["latitude"], data["results"][0]["longitude"]

def fetch_historical_data(city_name: str, years_back: int) -> pd.DataFrame:
    """Fetches historical hourly AQI AND Weather data for a specific city."""
    print(f"Fetching AQI & Weather data for {city_name.capitalize()}...")
    lat, lon = get_coordinates(city_name)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    # --- PULL AQI DATA ---
    url_aqi = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date.strftime('%Y-%m-%d')}"
        f"&end_date={end_date.strftime('%Y-%m-%d')}"
        f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone"
    )
    
    response_aqi = session.get(url_aqi, timeout=15)
    response_aqi.raise_for_status()
    data_aqi = response_aqi.json().get("hourly", {})
    
    df_aqi = pd.DataFrame({
        "timestamp": pd.to_datetime(data_aqi.get("time", [])),
        "pm10": data_aqi.get("pm10", []),
        "pm2_5": data_aqi.get("pm2_5", []),
        "no2": data_aqi.get("nitrogen_dioxide", []),
        "ozone": data_aqi.get("ozone", [])
    })

    # --- PULL WEATHER DATA ---
    url_weather = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date.strftime('%Y-%m-%d')}"
        f"&end_date={end_date.strftime('%Y-%m-%d')}"
        f"&hourly=temperature_2m,precipitation,wind_speed_10m"
    )
    
    response_weather = session.get(url_weather, timeout=15)
    response_weather.raise_for_status()
    data_weather = response_weather.json().get("hourly", {})
    
    df_weather = pd.DataFrame({
        "timestamp": pd.to_datetime(data_weather.get("time", [])),
        "temperature_2m": data_weather.get("temperature_2m", []),
        "precipitation": data_weather.get("precipitation", []),
        "wind_speed_10m": data_weather.get("wind_speed_10m", [])
    })

    # --- MERGE THEM TOGETHER ---
    # Combine the two datasets based on the exact hour they were recorded
    df_combined = pd.merge(df_aqi, df_weather, on="timestamp", how="inner")
    df_combined['city'] = city_name
    
    # Drop rows where sensors might have been offline, and return
    return df_combined.dropna().reset_index(drop=True)

def build_master_dataset() -> pd.DataFrame:
    """Iterates through global cities to build the combined dataset."""
    data_frames = []
    for city in GLOBAL_CITIES:
        try:
            df = fetch_historical_data(city, HISTORICAL_YEARS)
            data_frames.append(df)
            print(f"  -> Success for {city.capitalize()}! Resting for 3 seconds...")
            time.sleep(3) # Respect API limits between cities
        except Exception as e:
            print(f"  -> Error processing {city.capitalize()}: {e}")
            
    return pd.concat(data_frames, ignore_index=True)