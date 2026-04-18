import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from src.config import GLOBAL_CITIES, HISTORICAL_YEARS

def get_coordinates(city_name: str):
    """Resolves latitude and longitude for a given city name."""
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    if not data.get("results"):
        raise ValueError(f"Could not resolve coordinates for {city_name}")
        
    return data["results"][0]["latitude"], data["results"][0]["longitude"]

def fetch_historical_data(city_name: str, years_back: int) -> pd.DataFrame:
    """Fetches historical hourly AQI data for a specific city."""
    print(f"Fetching data for {city_name.capitalize()}...")
    lat, lon = get_coordinates(city_name)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years_back)
    
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date.strftime('%Y-%m-%d')}"
        f"&end_date={end_date.strftime('%Y-%m-%d')}"
        f"&hourly=pm10,pm2_5,nitrogen_dioxide,ozone"
    )
    
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get("hourly", {})
    
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data.get("time", [])),
        "pm10": data.get("pm10", []),
        "pm2_5": data.get("pm2_5", []),
        "no2": data.get("nitrogen_dioxide", []),
        "ozone": data.get("ozone", [])
    })
    
    df['city'] = city_name
    return df.dropna().reset_index(drop=True)

def build_master_dataset() -> pd.DataFrame:
    """Iterates through global cities to build the combined dataset."""
    data_frames = []
    for city in GLOBAL_CITIES:
        try:
            df = fetch_historical_data(city, HISTORICAL_YEARS)
            data_frames.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"Error processing {city}: {e}")
            
    return pd.concat(data_frames, ignore_index=True)

if __name__ == "__main__":
    
    print("data fetcher module testing")
    test_df = build_master_dataset()
    print(f"Success! Fetched {len(test_df)} rows.")