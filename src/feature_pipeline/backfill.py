from datetime import datetime, timedelta
from src.feature_pipeline.fetch_data import build_master_dataset
from src.feature_pipeline.compute_features import build_feature_pipeline
from src.feature_pipeline.store_features import upload_to_feature_store

def run():
    print("--- Starting Feature Pipeline (Incremental Update) ---")
    
    # Calculate a dynamic 14-day window
    # We fetch 14 days so that rolling averages and momentum features have enough context to calculate!
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Fetching data from {start_str} to {end_str}...")
    
    # 1. Extract raw data from Open-Meteo (passing the new dates!)
    raw_data = build_master_dataset(start_date=start_str, end_date=end_str)
    
    # 2. Transform into daily features and targets
    features = build_feature_pipeline(raw_data)
    
    # 3. Load into Hopsworks Feature Store
    upload_to_feature_store(features)
    
    print("--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    run()