import hopsworks
import pandas as pd
import time
from src.config import (
    HOPSWORKS_PROJECT_NAME, 
    FEATURE_GROUP_NAME, 
    FEATURE_GROUP_VERSION,
    HOPSWORKS_API_KEY
)

def upload_to_feature_store(features_df: pd.DataFrame):
    print("Authenticating with Hopsworks...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            project = hopsworks.login(
                host="eu-west.cloud.hopsworks.ai", 
                project=HOPSWORKS_PROJECT_NAME, 
                api_key_value=HOPSWORKS_API_KEY,
                hostname_verification=False
            )
            
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
            
            
            break 
            
        except Exception as e:
            print(f"\n[Warning] Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                print("Network hiccup detected. Waiting 15 seconds before retrying...")
                time.sleep(15)
            else:
                print("\n[Error] Max retries reached. Hopsworks is unreachable.")
                raise e 