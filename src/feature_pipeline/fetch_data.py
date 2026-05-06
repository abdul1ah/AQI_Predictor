import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.config import GLOBAL_CITIES

session = requests.Session()
retries = Retry(
    total=5,  
    backoff_factor=1, 
    status_forcelist=[429, 500, 502, 503, 504], 
)
session.mount('https://', HTTPAdapter(max_retries=retries))

def get_coordinates(city_name: str):
    """Resolves latitude and longitude for a given city name."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    
    response = session.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if not data.get("results"):
        raise ValueError(f"Could not resolve coordinates for {city_name}")
        
    return data["results"][0]["latitude"], data["results"][0]["longitude"]

def fetch_historical_data(city_name: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """Fetches Incremental AQI & Weather data, including the 'Synthetic Today' buffer."""
    print(f"Fetching AQI & Weather data for {city_name.capitalize()}...")
    lat, lon = get_coordinates(city_name)
    
    # We need to fetch one extra day into the future to grab forecast data for the rest of today
    end_date_obj = datetime.strptime(end_date_str, "%Y-%m-%d")
    fetch_end_date_str = (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')

    # --- 1. PULL AIR QUALITY DATA ---
    url_aqi = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date_str}"
        f"&end_date={fetch_end_date_str}"
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

    # --- 2. PULL WEATHER DATA (Forecast API supports recent history up to 3 months!) ---
    url_weather = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date_str}"
        f"&end_date={fetch_end_date_str}"
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

    # --- 3. COMBINE DATA ---
    df_master = pd.merge(df_aqi, df_weather, on="timestamp", how="inner")
    df_master['city'] = city_name
    
    # --- 4. THE SYNTHETIC TODAY FIX ---
    # Sort chronologically to ensure ffill works strictly forward in time
    df_master = df_master.sort_values("timestamp").reset_index(drop=True)
    
    # Forward-fill missing forecast data to bridge the gap between live analysis and future forecast
    df_master = df_master.ffill()
    
    # Slice off the "Tomorrow" buffer we fetched, ensuring the dataframe perfectly ends at the requested end_date
    df_master = df_master[df_master['timestamp'].dt.date <= pd.to_datetime(end_date_str).date()]

    # Safely drop any remaining NaNs
    return df_master.dropna().reset_index(drop=True)

def build_master_dataset(start_date: str, end_date: str) -> pd.DataFrame:
    """Iterates through global cities to build the combined dataset."""
    data_frames = []
    for city in GLOBAL_CITIES:
        try:
            # Pass the dynamic dates down to the fetcher!
            df = fetch_historical_data(city, start_date, end_date)
            data_frames.append(df)
            print(f"  -> Success for {city.capitalize()}! Resting for 3 seconds...")
            time.sleep(3) 
        except Exception as e:
            print(f"  -> Error processing {city.capitalize()}: {e}")
            
    return pd.concat(data_frames, ignore_index=True)