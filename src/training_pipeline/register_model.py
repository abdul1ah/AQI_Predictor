import os
import joblib
import hopsworks
from src.training_pipeline.fetch_training_data import get_training_dataset
from src.training_pipeline.train_evaluate import train_model
from src.config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY

def upload_models(trained_models_dict, metrics_dict):
    """Uploads the three champion multi-step models to Hopsworks."""
    print("Exporting model artifacts...")
    
    model_dir = "aqi_multi_step_models"
    os.makedirs(model_dir, exist_ok=True)
    
    for target, model in trained_models_dict.items():
        joblib.dump(model, os.path.join(model_dir, f"{target}_model.pkl"))

    print("Connecting to Hopsworks Model Registry...")
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai", 
        project=HOPSWORKS_PROJECT_NAME, 
        api_key_value=HOPSWORKS_API_KEY,
        hostname_verification=False
    )
    mr = project.get_model_registry()

    for target, metrics in metrics_dict.items():
        model_name = f"aqi_{target}_model"
        print(f"Registering {model_name}...")
        
        aqi_model = mr.python.create_model(
            name=model_name, 
            metrics=metrics,
            description=f"Champion model for {target} forecasting"
        )
        
        aqi_model.save(os.path.join(model_dir, f"{target}_model.pkl"))

if __name__ == "__main__":
    print("=== Starting Model Training Pipeline ===")
    
    data = get_training_dataset()
    
    print("Cleaning recent incomplete days from training data...")
    data = data.dropna().reset_index(drop=True)
    
    trained_models, eval_metrics = train_model(data)
    
    upload_models(trained_models, eval_metrics)
    
    print("=== Pipeline Complete ===")