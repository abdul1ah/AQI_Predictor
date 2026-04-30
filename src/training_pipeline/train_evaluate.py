import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import shap

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
        "Ridge_Regression": {
            "model": Ridge(), 
            "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [100, 300], 
                "learning_rate": [0.01, 0.05, 0.1], 
                "max_depth": [3, 5]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            "params": {
                "n_estimators": [100, 200], 
                "learning_rate": [0.05, 0.1], 
                "num_leaves": [31, 63],
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

        metrics = {
            "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
            "MAE": mean_absolute_error(y_test, predictions),
            "R2": r2_score(y_test, predictions)
        }
        
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        try:
            if best_target_name in ["Random_Forest", "XGBoost", "LightGBM"]:
                explainer = shap.TreeExplainer(best_target_model)
                shap_values = explainer.shap_values(X_test)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False)
                plt.savefig(f"shap_importance_{target}.png", bbox_inches='tight')
                plt.close()
                
            elif best_target_name == "Ridge_Regression":
                explainer = shap.LinearExplainer(best_target_model, X_train)
                shap_values = explainer.shap_values(X_test)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False)
                plt.savefig(f"shap_importance_{target}.png", bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"  SHAP bypassed: {e}")
            
        champion_models[target] = best_target_model
        all_metrics[target] = metrics

    print("\n--- All Multi-Step Models Trained Successfully ---")
    return champion_models, all_metrics