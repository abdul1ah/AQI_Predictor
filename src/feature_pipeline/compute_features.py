import pandas as pd
import numpy as np
import pandas as pd

def build_feature_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw hourly data into daily aggregated features and targets."""
    print("Initiating feature engineering pipeline...")
    
    # 1. Ensure datetime format and sort
    df = raw_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # 2. Daily Aggregation (Average pollution per day)
    daily_df = df.groupby(['city', 'date']).mean().reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values(by=['city', 'date']).reset_index(drop=True)
    
    # 3. Temporal Features (Crucial for XGBoost to understand Seasonality)
    print("  -> Extracting time-based features...")
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['day_of_year'] = daily_df['date'].dt.dayofyear
    
    # 4. Rolling Averages (Historical context)
    print("  -> Computing rolling averages...")
    daily_df['pm2_5_rolling_3d'] = daily_df.groupby('city')['pm2_5'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily_df['pm2_5_rolling_7d'] = daily_df.groupby('city')['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    
    print("  -> Computing AQI momentum (rate of change)...")
    daily_df['pm2_5_change_rate'] = daily_df.groupby('city')['pm2_5'].diff().fillna(0)
    
    # 6. Target Variables (What we want to predict: +1 day, +2 days)
    print("  -> Shifting targets for future forecasting...")
    daily_df['target_pm2_5_1d'] = daily_df.groupby('city')['pm2_5'].shift(-1)
    daily_df['target_pm2_5_2d'] = daily_df.groupby('city')['pm2_5'].shift(-2)
    daily_df['target_pm2_5_3d'] = daily_df.groupby('city')['pm2_5'].shift(-3)
    

    final_df = daily_df.dropna().reset_index(drop=True)
    
    final_df = final_df.drop(columns=['timestamp'], errors='ignore')
    
    print(f"Feature engineering complete. Matrix shape: {final_df.shape}")
    return final_df