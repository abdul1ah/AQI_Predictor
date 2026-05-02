from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import hopsworks
import joblib
import os
import pandas as pd
import requests
import asyncio
from datetime import datetime, timedelta

# Import your configurations
from src.config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY, FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION

# --- GLOBAL STATE (RAM) ---
app_state = {
    "mr": None,               # <-- Store Model Registry for hot swapping
    "models": {},
    "current_versions": {},  # <-- NEW: Track current model versions for comparison
    "feature_view": None,
    "batch_data": None,
    "cached_forecasts": {},
    "bg_task": None
}

targets = ["target_pm2_5_1d", "target_pm2_5_2d", "target_pm2_5_3d"]

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

def format_date_with_ordinal(date_obj):
    """Formats a datetime object to '2nd May 2026' format."""
    day = date_obj.day
    if 11 <= (day % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
    return f"{day}{suffix} {date_obj.strftime('%b %Y')}"

def fetch_live_current_aqi(city_name: str):
    """Hits Open-Meteo API to get the exact 'Right Now' AQI for the dashboard."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
        geo_data = requests.get(geo_url).json()
        if not geo_data.get("results"):
            return None
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        
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

# --- HYBRID CACHE & HOT SWAPPING LOGIC ---
def download_and_load_model(mr, target: str, version: int):
    """Downloads a specific model version (if missing) and loads it."""
    model_name = f"aqi_{target}_model"
    versioned_dir = f"{model_name}_v{version}"
    
    if not os.path.exists(versioned_dir):
        print(f"  -> Downloading {model_name} v{version} from cloud...")
        hw_model = mr.get_model(model_name, version=version)
        os.makedirs(versioned_dir, exist_ok=True)
        model_dir = hw_model.download(versioned_dir)
    else:
        print(f"  -> Cache Hit! Loading {model_name} v{version} directly from disk...")
        model_dir = versioned_dir
        
    # Load the new brain into RAM
    app_state["models"][target] = joblib.load(os.path.join(model_dir, f"{target}_model.pkl"))
    app_state["current_versions"][target] = version

def refresh_all_cache_and_models():
    """Blocking function to handle both Hot Swapping and Forecast Calculation."""
    print(f"[{datetime.now()}] --- Integrated Refresh Cycle Started ---")
    mr = app_state["mr"]

    # 1. Hot Swapping: Check for new model versions
    new_brains_detected = False
    for target in targets:
        model_name = f"aqi_{target}_model"
        
        # Poll Hopsworks for the actual max version
        print(f" -> Polling Registry for latest {model_name}...")
        all_models = mr.get_models(model_name)
        
        if not all_models:
            print(f"   ! Warning: Could not find any models for {model_name}. Skipping swap.")
            continue
            
        latest_version_on_registry = max([m.version for m in all_models])
        current_loaded_version = app_state["current_versions"].get(target, -1)

        # Comparison Logic
        if latest_version_on_registry > current_loaded_version:
            print(f" >>> HOT SWAP: Found new {model_name} v{latest_version_on_registry} (Currently loading v{current_loaded_version})")
            download_and_load_model(mr, target, latest_version_on_registry)
            new_brains_detected = True
    
    if new_brains_detected:
        print(" -> Hot Swapping Complete. Server is now using newer AI Brains.")
    else:
        print(" -> All local models are up to date.")

    # 2. Refesh Hopsworks Batch Data Matrix
    print(" -> Refreshing Hopsworks Batch Data...")
    batch_data = app_state["feature_view"].get_batch_data()
    app_state["batch_data"] = batch_data

    # 3. Forecast Pre-computation Calculation
    new_cache = {}
    cities = ["london", "beijing", "los angeles", "mumbai", "sydney", "delhi", "lahore", "karachi"]

    for c in cities:
        city_data = batch_data[batch_data['city'] == c].copy()
        if city_data.empty:
            continue

        time_col = 'date' if 'date' in city_data.columns else 'timestamp'
        city_data = city_data.sort_values(by=time_col)
        latest_row = city_data.tail(1)
        base_time = pd.to_datetime(latest_row[time_col].iloc[0])

        latest_data_point = latest_row.drop(columns=['city', 'date', 'timestamp'] + targets, errors='ignore')

        forecast_results = []
        for idx, target in enumerate(targets):
            days_ahead = idx + 1
            forecast_date = base_time + timedelta(days=days_ahead)

            # Use the newly swapped model directly from RAM
            model = app_state["models"][target]
            prediction_pm25 = max(0, float(model.predict(latest_data_point)[0]))

            forecast_results.append({
                "date": format_date_with_ordinal(forecast_date),
                "day_name": forecast_date.strftime('%A'),
                "predicted_aqi": pm25_to_aqi(prediction_pm25),
                "raw_pm25": round(prediction_pm25, 1)
            })

        new_cache[c] = {
            "base_data_date": base_time.strftime('%Y-%m-%d'),
            "predictions": forecast_results
        }

    app_state["cached_forecasts"] = new_cache
    print(f"[{datetime.now()}] --- Integrated Refresh Cycle Complete ---")

async def background_loop():
    """Background loop that executes the integrated refresh every hour."""
    while True:
        await asyncio.sleep(3600)  # Sleep for 1 hour
        try:
            # Execute blocking Hopsworks/Pandas logic in a separate thread
            await asyncio.to_thread(refresh_all_cache_and_models)
        except Exception as e:
            print(f"Error in background integrated refresh: {e}")

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
    app_state["mr"] = project.get_model_registry() # Store it here for hot swapping later

    print("2. Caching Feature View...")
    app_state["feature_view"] = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

    # Note: Initial Model download logic is now unified inside compute_all_forecasts for consistency.

    print("3. Initializing Integrated Pre-computation Cache (Hot Swap + Forecasts)...")
    # Loop manually here to handle initial model population before computation starts
    for target in targets:
        model_name = f"aqi_{target}_model"
        print(f" -> Booting latest {model_name}...")
        all_models = app_state["mr"].get_models(model_name)
        if not all_models:
             raise RuntimeError(f"Could not find any versions for {model_name}. Server boot aborted.")
        
        latest_version = max([m.version for m in all_models])
        download_and_load_model(app_state["mr"], target, latest_version)

    # Compute forecasts using the models just loaded above
    # Running forecasting logic *after* manual model loading above
    # Normally we call refresh_all_cache_and_models here, but we just manually handled the 
    # initial model load above. To be perfectly accurate on boot, we should do both.
    
    # We call this again to pre-fetch batch data and run initial forecasts.
    # We run in thread so boot remains asynchronous.
    await asyncio.to_thread(refresh_all_cache_and_models)

    print("4. Starting integrated background refresh loop...")
    app_state["bg_task"] = asyncio.create_task(background_loop())

    print("Server Boot Up Complete.")
    yield
    print("Shutting down server, clearing RAM...")
    if app_state["bg_task"]:
        app_state["bg_task"].cancel()
    app_state["models"].clear()
    app_state["current_versions"].clear()
    app_state["cached_forecasts"].clear()
    app_state["batch_data"] = None

app = FastAPI(title="AQI Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://aqi-predictor-two.vercel.app",
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
    Main endpoint. Returns dynamic live AQI + Pre-computed & Hot-swapped ML Forecast.
    """
    city = city.lower()
    
    # 1. Fetch 'Right Now' Live Data dynamically (0.1s)
    live_data = fetch_live_current_aqi(city)
    if not live_data:
        raise HTTPException(status_code=404, detail=f"Could not find live data for city: {city}")

    # 2. Grab the ML predictions from RAM (0ms)
    ml_data = app_state["cached_forecasts"].get(city)
    if not ml_data:
        raise HTTPException(status_code=404, detail=f"ML Forecast not available in cache for city: {city}")

    # 3. Construct Final Dashboard JSON Response
    return {
        "city": city.capitalize(),
        "status": "success",
        "current_live": {
            "aqi": live_data["aqi"],
            "raw_pm25": live_data["raw_pm25"],
            "note": "Fetched live via Open-Meteo"
        },
        "ml_forecast": ml_data
    }