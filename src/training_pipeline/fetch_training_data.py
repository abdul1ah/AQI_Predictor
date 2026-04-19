import hopsworks
import pandas as pd
from src.config import *

def get_training_dataset() -> pd.DataFrame:
    print("Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai", 
        project=HOPSWORKS_PROJECT_NAME, 
        api_key_value=HOPSWORKS_API_KEY,
        hostname_verification=False
    )
    fs = project.get_feature_store()
    
    # 1. Grab your Feature Group directly
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME, 
        version=FEATURE_GROUP_VERSION
    )
    
    # 2. Read the raw, unsplit dataframe (Bypassing Feature View stripping)
    print(f"Downloading raw dataset from {FEATURE_GROUP_NAME} V{FEATURE_GROUP_VERSION}...")
    df = fg.read()
    
    # Sort by date to prevent temporal data leakage (Great catch from your old code!)
    df = df.sort_values(by=['city', 'date']).reset_index(drop=True)
    
    print(f"Successfully downloaded {len(df)} rows.")
    return df