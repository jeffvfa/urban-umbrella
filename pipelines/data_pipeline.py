import pandas as pd
from pathlib import Path 
from config.settings import RAW_DATA_FILE, PROCESSED_DATA_DIR, PROCESSED_FEATURES_FILE

def preprocess(): 
    # Load the raw data
    df = pd.read_csv(RAW_DATA_FILE) 

    df['event_timestamp'] = pd.Timestamp.now()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True) #guarantee that the processed data directory exists
    
    # Save the processed data
    df.to_parquet(PROCESSED_FEATURES_FILE) 
    
    print(f"Data preprocessing complete. Processed data saved to {PROCESSED_FEATURES_FILE}") 
    
if __name__ == "__main__":
    preprocess()