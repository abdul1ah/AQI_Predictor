import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap

def evaluate_segmented(y_true: pd.Series, y_pred: np.ndarray, target_name: str) -> None:
    """Breaks down model accuracy based on the severity of the real-world pollution."""
    # Convert y_true to a numpy array to prevent pandas index misalignment with y_pred
    y_true_vals = y_true.values if isinstance(y_true, pd.Series) else y_true
    results = pd.DataFrame({'Actual': y_true_vals, 'Predicted': y_pred})
    
    # Define our real-world severity buckets
    bins = [-np.inf, 100, 200, np.inf]
    labels = ['Normal (<100)', 'Moderate (100-200)', 'Extreme (>200)']
    results['Severity'] = pd.cut(results['Actual'], bins=bins, labels=labels)
    
    print(f"\n--- SEGMENTED MAE FOR: {target_name.upper()} ---")
    
    for category in labels:
        subset = results[results['Severity'] == category]
        
        if len(subset) > 0:
            mae = mean_absolute_error(subset['Actual'], subset['Predicted'])
            pct = (len(subset) / len(results)) * 100
            print(f"[{category}]".ljust(20) + f" Count: {len(subset):<4} ({pct:>4.1f}%) | MAE: {mae:.2f}")
        else:
            print(f"[{category}]".ljust(20) + f" Count: 0    ( 0.0%) | MAE: N/A")
            
    print("-" * 50)

def train_model(train_data: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    Executes a multi-step training pipeline, building separate champion models 
    for 1-day, 2-day, and 3-day AQI forecasts.
    """
    print("Preparing data for multi-step training...")
    
    targets = ['target_pm2_5_1d', 'target_pm2_5_2d', 'target_pm2_5_3d']
    
    base_X = train_data.drop(columns=['city', 'date'] + targets)
    print("\n=== COLUMNS GIVEN TO THE MODEL ===")
    print(base_X.columns.tolist())
    print("=======================================================\n")
    
    model_zoo = {
        "Statistical_Ridge": {
            "model": Ridge(random_state=42), 
            "params": {
                "alpha": [0.1, 1.0, 10.0, 50.0]
            }
        },
        "Ensemble_RandomForest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [200, 400],
                "max_depth": [10, 20, None],
                "min_samples_leaf": [2, 5]
            }
        },
        "DeepLearning_MLP": {
            "model": MLPRegressor(random_state=42, early_stopping=True, max_iter=2000),
            "params": {
                "hidden_layer_sizes": [(128, 64, 32), (256, 128, 64)],
                "learning_rate_init": [0.001, 0.005],
                "alpha": [0.001, 0.01, 0.1]
            }
        },
        "GradientBoosting_XGBoost": {
            "model": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [200, 500], 
                "learning_rate": [0.01, 0.05], 
                "max_depth": [4, 6, 8],
                "subsample": [0.8],         
                "colsample_bytree": [0.8],  
                "reg_lambda": [1.0, 5.0]      
            }
        }
    }

    champion_models = {}
    all_metrics = {}
    
    for target in targets:
        print(f"\n{'='*50}")
        print(f"TRAINING PIPELINE FOR: {target.upper()}")
        print(f"{'='*50}")
        
        y = train_data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(base_X, y, test_size=0.2, shuffle=False)
        tscv = TimeSeriesSplit(n_splits=3)

        best_target_score = float('-inf')
        best_target_model = None
        best_target_name = ""

        for model_name, config in model_zoo.items():
            print(f"-> Cross-validating {model_name}...")
            grid_search = GridSearchCV(
                estimator=config["model"], param_grid=config["params"],
                cv=tscv, scoring='r2', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            if grid_search.best_score_ > best_target_score:
                best_target_score = grid_search.best_score_
                best_target_model = grid_search.best_estimator_
                best_target_name = model_name

        print(f"\n[WINNER FOR {target}]: {best_target_name}")

        predictions = best_target_model.predict(X_test)

        # Standard Metrics
        metrics = {
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "MAE": mean_absolute_error(y_test, predictions),
            "R2": r2_score(y_test, predictions)
        }
        
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        evaluate_segmented(y_test, predictions, target)

        try:
            # Tree-based models get TreeExplainer
            if best_target_name in ["Ensemble_RandomForest", "GradientBoosting_XGBoost"]:
                explainer = shap.TreeExplainer(best_target_model)
                shap_values = explainer.shap_values(X_test)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False)
                plt.savefig(f"shap_importance_{target}.png", bbox_inches='tight')
                plt.close()
                
            elif best_target_name == "Statistical_Ridge":
                explainer = shap.LinearExplainer(best_target_model, X_train)
                shap_values = explainer.shap_values(X_test)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False)
                plt.savefig(f"shap_importance_{target}.png", bbox_inches='tight')
                plt.close()
                
            elif best_target_name == "DeepLearning_MLP":
                print("  -> Generating SHAP for Deep Learning (Optimized with K-Means)...")
                background = shap.kmeans(X_train, 50)
                explainer = shap.KernelExplainer(best_target_model.predict, background)
                
                X_test_sampled = shap.sample(X_test, 100)
                shap_values = explainer.shap_values(X_test_sampled)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_sampled, show=False)
                plt.savefig(f"shap_importance_{target}.png", bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"  SHAP failed: {e}")
            
        champion_models[target] = best_target_model
        all_metrics[target] = metrics

    print("\n--- All Multi-Step Models Trained Successfully ---")
    return champion_models, all_metrics