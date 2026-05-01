from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import hopsworks
import joblib
import os
import pandas as pd
import requests
from datetime import datetime, timedelta

# Import your configurations
from src.config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY, FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION

# --- GLOBAL STATE (RAM) ---
# We store the models, feature view, and the data matrix here so they persist between API requests
app_state = {
    "models": {},
    "feature_view": None,
    "batch_data": None  # <-- Added for RAM caching
}

# --- HELPER FUNCTIONS ---
def pm25_to_aqi(pm25_value: float) -> int:
    """Converts raw PM2.5 to official AQI."""
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), 
        (35.5, 55.4, 101, 150), (55.5, 150.4, 151, 200), 
        (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)
    ]
    pm25_value = round(pm25_value, 1)
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= pm25_value <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25_value - c_low) + i_low
            return int(round(aqi))
    return 500

def fetch_live_current_aqi(city_name: str):
    """Hits Open-Meteo API to get the exact 'Right Now' AQI for the dashboard."""
    try:
        # 1. Geocode the city
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
        geo_data = requests.get(geo_url).json()
        if not geo_data.get("results"):
            return None
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        
        # 2. Fetch current hour AQI
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&current=pm2_5"
        aqi_data = requests.get(aqi_url).json()
        
        current_pm25 = aqi_data["current"]["pm2_5"]
        return {
            "raw_pm25": current_pm25,
            "aqi": pm25_to_aqi(current_pm25)
        }
    except Exception as e:
        print(f"Error fetching live AQI: {e}")
        return None

# --- SERVER LIFESPAN (Runs once on boot) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Booting FastAPI Server...")
    print("1. Connecting to Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai", 
        project=HOPSWORKS_PROJECT_NAME, 
        api_key_value=HOPSWORKS_API_KEY,
        hostname_verification=False
    )
    
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("2. Caching Feature View...")
    app_state["feature_view"] = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

    print("2.5 Pre-fetching ML Data into RAM...")
    # <-- Download the matrix ONCE when the server boots
    app_state["batch_data"] = app_state["feature_view"].get_batch_data()
    print(f" -> Successfully cached {len(app_state['batch_data'])} rows in memory.")

    print("3. Downloading and Loading Latest Models into RAM...")
    targets = ["target_pm2_5_1d", "target_pm2_5_2d", "target_pm2_5_3d"]
    
    for target in targets:
        model_name = f"aqi_{target}_model"
        
        # get the latest version number for this model from Hopsworks
        all_models = mr.get_models(model_name)
        latest_version = max([m.version for m in all_models])

        # Create a specific folder name that includes the version! (e.g., "aqi_target_pm2_5_1d_model_v18")
        versioned_dir = f"{model_name}_v{latest_version}"
        
        # BULLETPROOF LOGIC: Only download if this EXACT version is missing from your hard drive
        if not os.path.exists(versioned_dir):
            print(f"  -> Downloading {model_name} v{latest_version} from cloud...")
            hw_model = mr.get_model(model_name, version=latest_version)
            os.makedirs(versioned_dir, exist_ok=True)
            model_dir = hw_model.download(versioned_dir)
        else:
            print(f"  -> Cache Hit! Loading {model_name} v{latest_version} directly from disk...")
            model_dir = versioned_dir
            
        app_state["models"][target] = joblib.load(os.path.join(model_dir, f"{target}_model.pkl"))

    print("Server Boot Up Complete.")
    yield
    print("Shutting down server, clearing RAM...")
    app_state["models"].clear()
    app_state["batch_data"] = None

app = FastAPI(title="AQI Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "online", "message": "AQI Predictor is running."}

@app.get("/api/forecast")
def get_city_forecast(city: str = "karachi"):
    """
    Main endpoint. Returns live 'Right Now' AQI + 3-Day ML Forecast.
    """
    city = city.lower()
    
    # 1. Fetch 'Right Now' Live Data (No AI, just straight API)
    live_data = fetch_live_current_aqi(city)
    if not live_data:
        raise HTTPException(status_code=404, detail=f"Could not find live data for city: {city}")

    try:
        batch_data = app_state["batch_data"] 
        city_data = batch_data[batch_data['city'] == city].copy()
        
        if city_data.empty:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found in Machine Learning database.")
            
        # Sort and grab the absolute newest row for ML inference
        time_col = 'date' if 'date' in city_data.columns else 'timestamp'
        city_data = city_data.sort_values(by=time_col)
        latest_row = city_data.tail(1)
        base_time = pd.to_datetime(latest_row[time_col].iloc[0])
        
        # Drop metadata so Scikit-Learn doesn't crash
        targets = ["target_pm2_5_1d", "target_pm2_5_2d", "target_pm2_5_3d"]
        latest_data_point = latest_row.drop(columns=['city', 'date', 'timestamp'] + targets, errors='ignore')

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # 3. Run ML Inference (Instantaneous, models are in RAM)
    forecast_results = []
    
    for idx, target in enumerate(targets):
        days_ahead = idx + 1
        forecast_date = base_time + timedelta(days=days_ahead)

        model = app_state["models"][target]
        prediction_pm25 = model.predict(latest_data_point)[0]
        
        prediction_pm25 = max(0, float(prediction_pm25))
        
        forecast_results.append({
            "date": forecast_date.strftime('%Y-%m-%d'),
            "day_name": forecast_date.strftime('%A'),
            "predicted_aqi": pm25_to_aqi(prediction_pm25),
            "raw_pm25": round(prediction_pm25, 1)
        })

    # 4. Construct Final Dashboard JSON Response
    return {
        "city": city.capitalize(),
        "status": "success",
        "current_live": {
            "aqi": live_data["aqi"],
            "raw_pm25": live_data["raw_pm25"],
            "note": "Fetched live via Open-Meteo"
        },
        "ml_forecast": {
            "base_data_date": base_time.strftime('%Y-%m-%d'),
            "predictions": forecast_results
        }
    }