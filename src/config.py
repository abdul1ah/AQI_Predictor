import os
from dotenv import load_dotenv


load_dotenv()

HOPSWORKS_PROJECT_NAME = "aqi_prediction_project"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
FEATURE_GROUP_NAME = "global_aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "global_aqi_view"
FEATURE_VIEW_VERSION = 1

# cities to include in the AQI prediction model
GLOBAL_CITIES = [
    "london",      
    "beijing",     
    "los angeles", 
    "mumbai",      
    "sydney",
    "delhi",
    "lahore",
    "karachi"  
]
HISTORICAL_YEARS = 3