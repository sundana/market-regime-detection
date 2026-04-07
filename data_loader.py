import os
import pandas as pd



def load_tick_data(data_path):
    expected_years = list(range(2020, 2027))
    data_dict = {}
    print("Loading data from:", data_path)
    for year in expected_years:
        source_dir = os.path.join(data_path, f"{year}_combined.parquet")
        if os.path.exists(source_dir):
            data_dict[year] = pd.read_parquet(source_dir)
            print(f"Loaded {year}: {data_dict[year].shape[0]:,} rows")
        else:
            print(f"Failed to load {year}: File not found at {data_path}")
    
    print(f"Total file yang berhasil dimuat: {len(data_dict)} tahun")
    
    return data_dict



def preprocess_tick_data(data_ticks: dict):
    processed_ticks = {}
    for year, df in data_ticks.items():
        df.columns = ['timestamp', 'bid', 'ask']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['bid'] = pd.to_numeric(df['bid'])
        df['ask'] = pd.to_numeric(df['ask'])

        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        print(f"Preprocessing {year} selesai")
        processed_ticks[year] = df
    
    return processed_ticks



if __name__ == "__main__":
    file_path = "C:\\Users\\USER\\Documents\\Works\\NXVEST\\market-regime-detection\\data\\xauusd"
    ticks = load_tick_data(file_path)
    df = preprocess_tick_data(ticks)