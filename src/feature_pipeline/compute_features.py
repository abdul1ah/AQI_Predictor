import pandas as pd
import numpy as np

def build_feature_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw hourly data into daily aggregated features and targets."""
    print("Initiating feature engineering pipeline...")
    
    # 1. Ensure datetime format and sort
    df = raw_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    
    # 2. Daily Aggregation 
    daily_df = df.groupby(['city', 'date']).mean().reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values(by=['city', 'date']).reset_index(drop=True)
    
    # 3. Temporal Features 
    print("  -> Extracting time-based features...")
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['day_of_year'] = daily_df['date'].dt.dayofyear
    
    # --- NEW EXPERIMENTAL FEATURES: Cyclical Temporal Encoding ---
    print("  -> Encoding cyclical temporal features...")
    daily_df['month_sin'] = np.sin(2 * np.pi * daily_df['month'] / 12)
    daily_df['month_cos'] = np.cos(2 * np.pi * daily_df['month'] / 12)
    daily_df['day_of_week_sin'] = np.sin(2 * np.pi * daily_df['day_of_week'] / 7)
    daily_df['day_of_week_cos'] = np.cos(2 * np.pi * daily_df['day_of_week'] / 7)
    # -------------------------------------------------------------

    # 4. Rolling Averages
    print("  -> Computing rolling averages...")
    daily_df['pm2_5_rolling_3d'] = daily_df.groupby('city')['pm2_5'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    daily_df['pm2_5_rolling_7d'] = daily_df.groupby('city')['pm2_5'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # 5. Rate of Change
    print("  -> Computing AQI momentum (rate of change)...")
    daily_df['pm2_5_change_rate'] = daily_df.groupby('city')['pm2_5'].diff().fillna(0)
    
    # --- NEW EXPERIMENTAL FEATURES: Meteorological Interactions ---
    print("  -> Computing meteorological interactions...")
    # Stagnation proxy: High historical PM2.5 combined with low wind speed.
    # Added 0.1 to wind_speed_10m to safely prevent division by zero errors.
    if 'wind_speed_10m' in daily_df.columns:
        daily_df['stagnation_index'] = daily_df['pm2_5_rolling_3d'] / (daily_df['wind_speed_10m'] + 0.1)
    # --------------------------------------------------------------
    
    # 6. Target Variables
    print("  -> Shifting targets for future forecasting...")
    daily_df['target_pm2_5_1d'] = daily_df.groupby('city')['pm2_5'].shift(-1)
    daily_df['target_pm2_5_2d'] = daily_df.groupby('city')['pm2_5'].shift(-2)
    daily_df['target_pm2_5_3d'] = daily_df.groupby('city')['pm2_5'].shift(-3)
    
    # Dynamically grab all columns that aren't targets to use as inputs
    input_cols = [col for col in daily_df.columns if not col.startswith('target_')]

    # Drop rows where input features have NaNs (due to rolling averages/lags)
    final_df = daily_df.dropna(subset=input_cols).reset_index(drop=True)
    
    final_df = final_df.drop(columns=['timestamp'], errors='ignore')
    
    print(f"Feature engineering complete. Matrix shape: {final_df.shape}")
    return final_df