import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def build_feature_pipeline(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw hourly data into daily aggregated statistical features and targets, while filtering hardware glitches."""
    print("Initiating smart feature engineering pipeline...")
 
    # 1. Ensure datetime format and sort
    df = raw_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort chronologically by city and time so our interpolation and diffs work correctly
    df = df.sort_values(by=['city', 'timestamp']).reset_index(drop=True)
    df['date'] = df['timestamp'].dt.date
 
    # 1.5. Detect and Repair Isolated Sensor Glitches (Hourly Level)
    print("  -> Scanning for isolated sensor hardware glitches...")
    
    # a glitch being a massive, physically impossible 1-hour spike
    # Example: Spiking UP by more than 300, and immediately crashing DOWN by more than 300 in the next hour
    for col in ['pm2_5', 'pm10']:
        if col in df.columns:
            # diff(1) is Current - Previous. diff(-1) is Current - Next.
            diff_prev = df.groupby('city')[col].diff(1)
            diff_next = df.groupby('city')[col].diff(-1)
            
            # Identify rows that are massive isolated peaks
            is_glitch = (diff_prev > 300) & (diff_next > 300)
            glitch_count = is_glitch.sum()
            
            if glitch_count > 0:
                print(f"     [!] WARNING: Found and neutralized {glitch_count} hardware glitches in {col}.")
                # Replace the fake massive spike with NaN, then linearly interpolate the gap
                df.loc[is_glitch, col] = np.nan
                df[col] = df.groupby('city')[col].transform(lambda x: x.interpolate(method='linear', limit_direction='both'))

    # 2. Daily Aggregation (The Smart Way)
    print("  -> Aggregating hourly data into daily statistical features...")
    
    aggregation_rules = {
        'pm2_5': ['mean', 'max', 'std'],         # Average, worst spike, and volatility
        'pm10': ['mean', 'max'],
        'no2': ['mean'],
        'ozone': ['mean', 'max'],
        'temperature_2m': ['mean', 'max', 'min'], # Daily high and low temps
        'precipitation': ['sum'],                 # Total rain for the day
        'wind_speed_10m': ['mean', 'max']         # Average wind, and maximum gusts
    }
    
    daily_df = df.groupby(['city', 'date']).agg(aggregation_rules).reset_index()
 
    # Flatten the multi-level columns created by .agg()
    daily_df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in daily_df.columns]
 
    # Rename our base PM2.5 mean back to just 'pm2_5' to keep downstream logic intact
    daily_df = daily_df.rename(columns={'pm2_5_mean': 'pm2_5'})
 
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df = daily_df.sort_values(by=['city', 'date']).reset_index(drop=True)
    
    # Safety catch: Standard Deviation (std) returns NaN if a day only had 1 hour of data. 
    daily_df = daily_df.fillna(0)
 
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