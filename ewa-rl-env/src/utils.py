# Feature Engineering Functions

import pandas as pd
import numpy as np

# Price Intervals
def calculate_price_intervals(df, price_col='token0Price'):
    df['price_high'] = df[price_col].rolling(window=10, min_periods=1).max() # 10 period high
    df['price_low'] = df[price_col].rolling(window=10, min_periods=1).min() # 10 period low
    df['price_range'] = df['price_high'] - df['price_low'] # Price range
    return df

# Volatility Calculation
def calculate_volatility(df, price_col='token0Price'):
    df['price_change'] = df[price_col].pct_change() # Price change %
    df['volatility'] = df['price_change'].rolling(window=10, min_periods=1).std() # 10 period rolling volatility
    return df

# Volume Features
def calculate_volume_averages(df, volume_col='volumeUSD'):
    df['volume_avg_10'] = df[volume_col].rolling(window=10, min_periods=1).mean() # 10 period average volume
    df['volume_avg_30'] = df[volume_col].rolling(window=30, min_periods=1).mean() # 30 period average volume
    return df

# Liquidity and Fees Features
def add_liquidity_and_fees(df, liquidity_col='liquidity', fees_col='feesUSD'):
    df['liquidity'] = pd.to_numeric(df[liquidity_col], errors='coerce')
    df['feesUSD'] = pd.to_numeric(df[fees_col], errors='coerce')
    return df

# Time Features
def add_time_features(df, timestamp_col='timestamp'):
    df['hour'] = df[timestamp_col].dt.hour
    df['day'] = df[timestamp_col].dt.day
    df['month'] = df[timestamp_col].dt.month
    df['year'] = df[timestamp_col].dt.year
    df['day_of_week'] = df[timestamp_col].dt.dayofweek
    return df

# Combine all feature engineering functions
def engineer_features(df):
    columns_to_convert = ['token0Price', 'volumeUSD', 'liquidity', 'feesUSD']
    
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    if 'token0Price' in df.columns:
        df = calculate_price_intervals(df)
        df = calculate_volatility(df)

    if 'volumeUSD' in df.columns:
        df = calculate_volume_averages(df)

    if 'liquidity' in df.columns and 'feesUSD' in df.columns:
        df = add_liquidity_and_fees(df)

    if 'price' in df.columns:
        df = calculate_volatility(df, price_col='price')

    if 'timestamp' in df.columns:
        df = add_time_features(df)

    df.dropna()
    
    return df

