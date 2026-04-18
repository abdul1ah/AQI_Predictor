from src.feature_pipeline.fetch_data import build_master_dataset
from src.feature_pipeline.compute_features import build_feature_pipeline
from src.feature_pipeline.store_features import upload_to_feature_store

def run():
    print("--- Starting Feature Pipeline Backfill ---")
    
    # 1. Extract raw data from Open-Meteo
    raw_data = build_master_dataset()
    
    # 2. Transform into daily features and targets
    features = build_feature_pipeline(raw_data)
    
    # 3. Load into Hopsworks Feature Store
    upload_to_feature_store(features)
    
    print("--- Pipeline Execution Finished ---")

if __name__ == "__main__":
    run()