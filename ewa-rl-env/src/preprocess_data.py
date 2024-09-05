import os
import json
import pandas as pd

# Paths for the data
DATA_DIR = '../data/'
FILE_STRUCTURE = {
    'burn': ['b-ETH-USDC-high.txt', 'b-ETH-USDC-low.txt', 'b-USDT-USDC-high.txt', 'b-USDT-USDC-low.txt'],
    'daily-pool': ['dp-ETH-USDC-high.txt', 'dp-ETH-USDC-low.txt', 'dp-USDT-USDC-high.txt', 'dp-USDT-USDC-low.txt'],
    'historical-swaps': ['hs-ETH-USDC-high.txt', 'hs-ETH-USDC-low.txt', 'hs-USDT-USDC-high.txt', 'hs-USDT-USDC-low.txt'],
    'hourly-pool': ['hp-ETH-USDC-high.txt', 'hp-ETH-USDC-low.txt', 'hp-USDT-USDC-high.txt', 'hp-USDT-USDC-low.txt'],
    'mint': ['m-ETH-USDC-high.txt', 'm-ETH-USDC-low.txt', 'm-USDT-USDC-high.txt', 'm-USDT-USDC-low.txt']
}

def load_json_file(file_path):
    # Load JSON data from a file
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def parse_data_from_directory(data_type):
    # Parse all files from a specific directory (data_type)
    data = []
    for file_name in FILE_STRUCTURE[data_type]:
        file_path = os.path.join(DATA_DIR, data_type, file_name)
        file_data = load_json_file(file_path)
        data.append(file_data['data'])
    return data

def parse_swaps_data():
    # Parse swaps data from historical-swaps files
    swaps_data = parse_data_from_directory('historical-swaps')
    return swaps_data

def parse_mints_data():
    # Parse mints data from mint files
    mints_data = parse_data_from_directory('mint')
    return mints_data

def parse_burns_data():
    # Parse burns data from burn files
    burns_data = parse_data_from_directory('burn')
    return burns_data

def parse_daily_pool_data():
    # Parse daily pool data
    daily_pool_data = parse_data_from_directory('daily-pool')
    return daily_pool_data

def parse_hourly_pool_data():
    # Parse hourly pool data
    hourly_pool_data = parse_data_from_directory('hourly-pool')
    return hourly_pool_data

def preprocess_all_data():
    swaps = parse_swaps_data()
    mints = parse_mints_data()
    burns = parse_burns_data()
    daily_pool = parse_daily_pool_data()
    hourly_pool = parse_hourly_pool_data()
    
    return {
        'swaps': swaps,
        'mints': mints,
        'burns': burns,
        'daily_pool': daily_pool,
        'hourly_pool': hourly_pool
    }

# Run preprocessing
preprocessed_data = preprocess_all_data()
print("Data preprocessing complete.")
print("\n")

# Convert timestamps method
def convert_timestamps(df, timestamp_col='timestamp'):
    # Rename 'date' or 'periodStartUnix' columns to 'timestamp' for consistency
    if 'date' in df.columns:
        df.rename(columns={'date': 'timestamp'}, inplace=True)
    elif 'periodStartUnix' in df.columns:
        df.rename(columns={'periodStartUnix': 'timestamp'}, inplace=True)

    # Convert timestamp column to datetime
    df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors='coerce')
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
    return df

def convert_to_dataframe(data, columns, key='swaps'):
    # Convert list of dictionaries to a pandas DataFrame. 'key' specifies which key to extract from each file 
    flattened_data = []
    for file_data in data:
        if key:
            file_data = file_data[key] 
        flattened_data.extend(file_data) 
    return pd.DataFrame(flattened_data, columns=columns)



# Define columns for each dataset
swaps_columns = ['amount0', 'amount1', 'id', 'logIndex', 'sqrtPriceX96', 'tick', 'timestamp']
mints_columns = ['amount', 'amount0', 'amount1', 'amountUSD', 'id', 'timestamp']
burns_columns = ['amount', 'amount0', 'amount1', 'amountUSD', 'id', 'timestamp']
daily_pool_df_columns = ['date', 'feesUSD', 'liquidity', 'sqrtPrice', 'token0Price', 'token1Price', 'tvlUSD', 'txCount', 'volumeUSD']
hourly_pool_df_columns = ['feesUSD', 'liquidity', 'periodStartUnix', 'sqrtPrice', 'token0Price', 'token1Price', 'tvlUSD', 'txCount', 'volumeUSD']

# Create DataFrames
swaps_df = convert_to_dataframe(preprocessed_data['swaps'], swaps_columns, key='swaps')
mints_df = convert_to_dataframe(preprocessed_data['mints'], mints_columns, key='mints')
burns_df = convert_to_dataframe(preprocessed_data['burns'], burns_columns, key='burns')
daily_pool_df = convert_to_dataframe(preprocessed_data['daily_pool'], daily_pool_df_columns, key='poolDayDatas')
hourly_pool_df = convert_to_dataframe(preprocessed_data['hourly_pool'], hourly_pool_df_columns, key='poolHourDatas')

# Convert timestamps
swaps_df = convert_timestamps(swaps_df)
mints_df = convert_timestamps(mints_df)
burns_df = convert_timestamps(burns_df)
daily_pool_df = convert_timestamps(daily_pool_df)
hourly_pool_df = convert_timestamps(hourly_pool_df)

# Display the first few rows of swaps data
print("Swaps data head:")
print(swaps_df.head())
print("\n")
print("Mints data head:")
print(mints_df.head())
print("\n")
print("Burns data head:")
print(burns_df.head())
print("\n")
print("Daily pool data head:")
print(daily_pool_df.head())
print("\n")
print("Hourly pool data head:")
print(hourly_pool_df.head())
print("\n")