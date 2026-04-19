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

def train_model(train_data: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    """
    Executes a comprehensive training pipeline including Time Series Cross-Validation,
    hyperparameter tuning across advanced models, and SHAP feature evaluation.
    """
    print("Preparing data for training...")
    
    # Isolate features (X) and the primary 1-day target (y)
    X = train_data.drop(columns=['city', 'date', 'target_pm2_5_1d', 'target_pm2_5_2d', 'target_pm2_5_3d'])
    y = train_data['target_pm2_5_1d']

    # Sequential split (80% train, 20% test) to respect time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define the Time Series cross-validation strategy
    tscv = TimeSeriesSplit(n_splits=3)

    # Expanded Model Zoo including XGBoost and LightGBM
    model_zoo = {
        "Ridge_Regression": {
            "model": Ridge(),
            "params": {"alpha": [0.1, 1.0, 10.0]}
        },
        "Random_Forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10]
            }
        },
        "Gradient_Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 50]
            }
        }
    }

    best_overall_model = None
    best_overall_score = float('-inf')
    best_model_name = ""

    print("\nInitiating GridSearchCV across expanded model zoo...")
    
    for model_name, config in model_zoo.items():
        print(f"Running grid search for {model_name}...")
        
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=tscv,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Validation R2: {grid_search.best_score_:.4f}")
        
        if grid_search.best_score_ > best_overall_score:
            best_overall_score = grid_search.best_score_
            best_overall_model = grid_search.best_estimator_
            best_model_name = model_name

    print("\n--- Hyperparameter Tuning Complete ---")
    print(f"Selected Champion Model: {best_model_name}")

    print("\nEvaluating champion model on holdout test set...")
    predictions = best_overall_model.predict(X_test)
    metrics = {
        "RMSE": mean_squared_error(y_test, predictions, squared=False),
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions)
    }

    print("\n--- Final Model Evaluation Metrics ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("--------------------------------------\n")

    # --- SHAP FEATURE IMPORTANCE ---
    print("Generating SHAP Feature Explanations...")
    try:
        # SHAP TreeExplainer is highly optimized for RF, GB, XGBoost, and LightGBM
        if best_model_name in ["Random_Forest", "Gradient_Boosting", "XGBoost", "LightGBM"]:
            explainer = shap.TreeExplainer(best_overall_model)
            shap_values = explainer.shap_values(X_test)
            
            # Save the plot so we can view it later
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig("shap_feature_importance.png", bbox_inches='tight')
            print("SHAP summary plot successfully saved as 'shap_feature_importance.png'.")
        else:
            print("Champion model is linear; skipping SHAP TreeExplainer.")
    except Exception as e:
        print(f"SHAP generation bypassed: {e}")

    return best_overall_model, metrics