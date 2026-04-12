import os
import time
import pandas as pd



def load_tick_data(data_path):
    expected_years = list(range(2020, 2027))
    data_dict = {}
    print("Loading data from:", data_path)
    started_at = time.perf_counter()
    total_years = len(expected_years)

    for idx, year in enumerate(expected_years, start=1):
        print(f"[Data Load] {idx}/{total_years} - Reading year {year}...")
        source_dir = os.path.join(data_path, f"{year}_combined.parquet")
        if os.path.exists(source_dir):
            data_dict[year] = pd.read_parquet(source_dir)
            print(f"Loaded {year}: {data_dict[year].shape[0]:,} rows")
        else:
            print(f"Failed to load {year}: File not found at {data_path}")
    
    elapsed = time.perf_counter() - started_at
    print(f"Total file yang berhasil dimuat: {len(data_dict)} tahun")
    print(f"[Data Load] Completed in {elapsed:.1f}s")
    
    return data_dict



def preprocess_tick_data(data_ticks: dict):
    processed_ticks = {}
    total = len(data_ticks)
    for idx, (year, df) in enumerate(data_ticks.items(), start=1):
        df.columns = ['timestamp', 'bid', 'ask']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['bid'] = pd.to_numeric(df['bid'])
        df['ask'] = pd.to_numeric(df['ask'])

        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        print(f"[Preprocess] {idx}/{total} year={year} selesai")
        processed_ticks[year] = df
    
    return processed_ticks



if __name__ == "__main__":
    file_path = "C:\\Users\\USER\\Documents\\Works\\NXVEST\\market-regime-detection\\data\\xauusd"
    ticks = load_tick_data(file_path)
    df = preprocess_tick_data(ticks)