import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_feature_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw hourly data into daily aggregated features and targets."""
    print("Initiating feature engineering pipeline...")
 
    # 1. Ensure datetime format and sort
    df = raw_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
 
    # 2. Daily Aggregation
    daily_df = df.groupby(['city', 'date']).mean(numeric_only=True).reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values(by=['city', 'date']).reset_index(drop=True)
 
    # 3. Temporal Features
    print("  -> Extracting time-based features...")
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['day_of_year'] = daily_df['date'].dt.dayofyear
 
    # 4. Rolling Averages
    print("  -> Computing rolling averages...")
    daily_df['pm2_5_rolling_3d'] = daily_df.groupby('city')['pm2_5'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily_df['pm2_5_rolling_7d'] = daily_df.groupby('city')['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
 
    # 5. Rate of Change
    print("  -> Computing AQI momentum (rate of change)...")
    daily_df['pm2_5_change_rate'] = daily_df.groupby('city')['pm2_5'].diff().fillna(0)
 
    # 6. Target Variables
    print("  -> Shifting targets for future forecasting...")
    daily_df['target_pm2_5_1d'] = daily_df.groupby('city')['pm2_5'].shift(-1)
    daily_df['target_pm2_5_2d'] = daily_df.groupby('city')['pm2_5'].shift(-2)
    daily_df['target_pm2_5_3d'] = daily_df.groupby('city')['pm2_5'].shift(-3)
 
    # 7. Safety Cleanup (Fixing the Hopsworks crashes)
    input_cols = [col for col in daily_df.columns if not col.startswith('target_')]
    final_df = daily_df.dropna(subset=input_cols).reset_index(drop=True)
    
    # Fill target NaNs with a dummy float to prevent Hopsworks PySpark schema crashes
    target_cols = [col for col in daily_df.columns if col.startswith('target_')]
    final_df[target_cols] = final_df[target_cols].fillna(-1.0)
 
    final_df = final_df.drop(columns=['timestamp'], errors='ignore')
    
    # 8. THE INCREMENTAL SLICE (Push Narrow)
    # We only want to push the last 2 days to Hopsworks to save RAM and prevent overwriting good history
    cutoff_date = pd.to_datetime(datetime.now().date() - timedelta(days=2))
    incremental_df = final_df[final_df['date'] >= cutoff_date].copy()
 
    print(f"Feature engineering complete. Matrix shape reduced from {final_df.shape} to {incremental_df.shape} for incremental upload.")
    return incremental_df