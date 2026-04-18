import pandas as pd
import numpy as np

def build_feature_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw hourly data into a daily feature set."""
    print("Initiating feature engineering pipeline...")
    df = raw_df.copy()
    
    df['date'] = df['timestamp'].dt.date
    
    daily_df = df.groupby(['city', 'date']).agg({
        'pm10': 'mean',
        'pm2_5': 'mean',
        'no2': 'mean',
        'ozone': 'mean'
    }).reset_index()
    
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values(by=['city', 'date']).reset_index(drop=True)
    
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    
    grouped = daily_df.groupby('city')
    daily_df['pm2_5_change_rate'] = grouped['pm2_5'].diff()
    daily_df['pm2_5_rolling_3d'] = grouped['pm2_5'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily_df['pm2_5_rolling_7d'] = grouped['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    daily_df['target_pm2_5_1d'] = grouped['pm2_5'].shift(-1)
    daily_df['target_pm2_5_2d'] = grouped['pm2_5'].shift(-2)
    daily_df['target_pm2_5_3d'] = grouped['pm2_5'].shift(-3)
    
    # handle missing/null values by dropping rows with any null values
    final_df = daily_df.dropna().reset_index(drop=True)
    
    print(f"Feature engineering complete. Matrix shape: {final_df.shape}")
    return final_df