import hopsworks
import pandas as pd
from src.config import *

def get_training_dataset() -> pd.DataFrame:
    print("1. Connecting to Hopsworks Feature Store...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai", 
        project=HOPSWORKS_PROJECT_NAME, 
        api_key_value=HOPSWORKS_API_KEY,
        hostname_verification=False
    )
    fs = project.get_feature_store()
    
    print(f"2. Fetching Feature Group: {FEATURE_GROUP_NAME} V{FEATURE_GROUP_VERSION}...")
    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME, 
        version=FEATURE_GROUP_VERSION
    )
    
    print(f"3. Validating/Creating Feature View: {FEATURE_VIEW_NAME} V{FEATURE_VIEW_VERSION}...")
    query = fg.select_all()
    
    try:
        feature_view = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    except Exception:
        feature_view = None
        
    if feature_view is None:
        print(" -> Feature View not found. Creating it now...")
        feature_view = fs.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            query=query
        )
        print(" -> Feature View created successfully!")
    else:
        print(" -> Feature View already exists. Proceeding...")
    
    print("4. Downloading dataset for training...")
    df = query.read()
    
    df = df.sort_values(by=['city', 'date']).reset_index(drop=True)
    
    print(f"Successfully downloaded {len(df)} rows.")
    return df