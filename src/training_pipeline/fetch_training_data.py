import hopsworks
import pandas as pd
from src.config import (
    HOPSWORKS_PROJECT_NAME,
    HOPSWORKS_API_KEY, 
    FEATURE_GROUP_NAME, 
    FEATURE_VIEW_NAME, 
    FEATURE_VIEW_VERSION
)

def get_training_dataset() -> pd.DataFrame:
    """Fetches the engineered feature matrix from Hopsworks."""
    print("Connecting to Hopsworks Feature Store...")
    

    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai", 
        project=HOPSWORKS_PROJECT_NAME, 
        api_key_value=HOPSWORKS_API_KEY,
        hostname_verification=False      
    )
    
    # THIS IS THE LINE THAT WAS MISSING
    fs = project.get_feature_store()

    print("Retrieving Feature Group...")
    aqi_fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=1)
    query = aqi_fg.select_all()

    print(f"Creating/Retrieving Feature View '{FEATURE_VIEW_NAME}'...")
    feature_view = fs.get_or_create_feature_view(
        name=FEATURE_VIEW_NAME,
        version=FEATURE_VIEW_VERSION,
        description="Dataset for AQI forecasting",
        labels=["target_pm2_5_1d", "target_pm2_5_2d", "target_pm2_5_3d"],
        query=query
    )

    print("Downloading training dataset...")
    train_data, _ = feature_view.training_data(description="AQI baseline training data")

    # Sort by date to prevent temporal data leakage during the train/test split
    train_data = train_data.sort_values(by='date').reset_index(drop=True)
    return train_data