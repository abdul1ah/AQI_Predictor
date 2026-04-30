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
    """Fetches historical hourly AQI AND Weather data for a specific city, bridging archive and live APIs."""
    print(f"Fetching AQI & Weather data for {city_name.capitalize()}...")
    lat, lon = get_coordinates(city_name)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    recent_start_date = end_date - timedelta(days=5)

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

    # --- 2. PULL WEATHER DATA: DEEP HISTORY (Archive API) ---
    url_weather_archive = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date.strftime('%Y-%m-%d')}"
        f"&end_date={(recent_start_date - timedelta(days=1)).strftime('%Y-%m-%d')}"
        f"&hourly=temperature_2m,precipitation,wind_speed_10m"
    )
    
    response_archive = session.get(url_weather_archive, timeout=15)
    response_archive.raise_for_status()
    data_archive = response_archive.json().get("hourly", {})
    
    df_weather_archive = pd.DataFrame({
        "timestamp": pd.to_datetime(data_archive.get("time", [])),
        "temperature_2m": data_archive.get("temperature_2m", []),
        "precipitation": data_archive.get("precipitation", []),
        "wind_speed_10m": data_archive.get("wind_speed_10m", [])
    })

    # --- 3. PULL WEATHER DATA: RECENT & LIVE (Forecast API) ---
    url_weather_recent = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={recent_start_date.strftime('%Y-%m-%d')}"
        f"&end_date={end_date.strftime('%Y-%m-%d')}"
        f"&hourly=temperature_2m,precipitation,wind_speed_10m"
    )
    
    response_recent = session.get(url_weather_recent, timeout=15)
    response_recent.raise_for_status()
    data_recent = response_recent.json().get("hourly", {})
    
    df_weather_recent = pd.DataFrame({
        "timestamp": pd.to_datetime(data_recent.get("time", [])),
        "temperature_2m": data_recent.get("temperature_2m", []),
        "precipitation": data_recent.get("precipitation", []),
        "wind_speed_10m": data_recent.get("wind_speed_10m", [])
    })

    df_weather_combined = pd.concat([df_weather_archive, df_weather_recent], ignore_index=True)
    
    df_weather_combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)

    df_master = pd.merge(df_aqi, df_weather_combined, on="timestamp", how="inner")
    df_master['city'] = city_name
    
    return df_master.dropna().reset_index(drop=True)

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