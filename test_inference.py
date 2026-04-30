import hopsworks
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
from src.config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY, FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION

def test_live_predictions(target_city="delhi"):
    print("1. Connecting to Hopsworks...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai", 
        project=HOPSWORKS_PROJECT_NAME, 
        api_key_value=HOPSWORKS_API_KEY,
        hostname_verification=False
    )
    
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print(f"2. Fetching today's engineered data from Feature View: {FEATURE_VIEW_NAME} (Version {FEATURE_VIEW_VERSION})...")
    
    try:
        feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    except Exception as e:
        print(f"\nCRITICAL ERROR: Could not find Feature View '{FEATURE_VIEW_NAME}' Version {FEATURE_VIEW_VERSION}.")
        print("FIX: You need to run your Model Training Pipeline first so it can create the Feature View!")
        return
    
    # Fetch only the last 48 hours to save memory and time
    today = datetime.now()
    yesterday = today - timedelta(days=2)
    
    batch_data = feature_view.get_batch_data(
        start_time=yesterday,
        end_time=today
    )
    
    # Filter down to just the requested city
    city_data = batch_data[batch_data['city'] == target_city]
    
    if city_data.empty:
        print(f"Error: No recent data found for {target_city.capitalize()}.")
        return

    # Grab the single latest row and drop the metadata columns
    latest_data_point = city_data.tail(1).drop(columns=['city', 'date'], errors='ignore')

    print(f"3. Running Models for {target_city.capitalize()}...")
    targets = ["target_pm2_5_1d", "target_pm2_5_2d", "target_pm2_5_3d"]
    
    for target in targets:
        model_name = f"aqi_{target}_model"
        try:
            # Fetch the model from the registry
            hw_model = mr.get_model(model_name, version=1)
            model_dir = hw_model.download()
            loaded_model = joblib.load(os.path.join(model_dir, f"{target}_model.pkl"))
            
            # Predict!
            prediction = loaded_model.predict(latest_data_point)[0]
            print(f" -> Forecast for {target}: {prediction:.2f} µg/m³ PM2.5")
        except Exception as e:
            print(f" -> Error running {model_name}: {e}")

if __name__ == "__main__":
    # You can change this to "london", "beijing", etc.
    test_live_predictions(target_city="delhi")