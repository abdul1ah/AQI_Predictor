import os
import joblib
import hopsworks
from src.config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY
from src.training_pipeline.fetch_training_data import get_training_dataset
from src.training_pipeline.train_evaluate import train_model

def upload_model(model, metrics: dict):
    """Packages the model and registers it to the Hopsworks cloud."""
    print("Exporting model artifact...")
    model_dir = "aqi_model_dir"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/random_forest_aqi.pkl")

    print("Connecting to Hopsworks Model Registry...")
    
    # Explicitly disabling hostname verification
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai", 
        project=HOPSWORKS_PROJECT_NAME, 
        api_key_value=HOPSWORKS_API_KEY,
        hostname_verification=False
    )
    
    mr = project.get_model_registry()

    print("Registering model artifact...")
    aqi_model = mr.python.create_model(
        name="aqi_forecasting_model",
        metrics=metrics,
        description="Random Forest regressor predicting next-day PM2.5 levels."
    )

    aqi_model.save(model_dir)
    print("Model successfully registered in the cloud.")

if __name__ == "__main__":
    print("=== Starting Model Training Pipeline ===")
    
    # 1. Fetch Data
    data = get_training_dataset()
    
    # 2. Train and Evaluate
    trained_model, eval_metrics = train_model(data)
    
    # 3. Register to Cloud
    upload_model(trained_model, eval_metrics)
    
    print("=== Pipeline Execution Finished ===")