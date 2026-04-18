import hopsworks
import pandas as pd
from src.config import (
    HOPSWORKS_PROJECT_NAME, 
    FEATURE_GROUP_NAME, 
    FEATURE_GROUP_VERSION
)

def upload_to_feature_store(features_df: pd.DataFrame):
    """Uploads the engineered feature matrix to Hopsworks."""
    print("Authenticating with Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME)
    fs = project.get_feature_store()
    
    print(f"Configuring Feature Group: {FEATURE_GROUP_NAME}...")
    aqi_fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["city", "date"],
        description="Daily aggregated AQI features and multi-day targets for global cities."
    )
    
    print("Uploading feature matrix to the Feature Store. This may take a few minutes...")
    aqi_fg.insert(features_df)
    print("Feature upload complete. Data is now securely stored in the cloud.")