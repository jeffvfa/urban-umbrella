from pathlib import Path

# Define paths for raw and processed data 
BASE_DIR = Path(__file__).resolve().parents[1] # Get the parent directory of the current file (config) and then go one level up to the project root

# data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" 
PROCESSED_DATA_DIR = DATA_DIR / "processed"  

# data files 
RAW_DATA_FILE = RAW_DATA_DIR / "data.csv" 
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "features.parquet" 

# mlflow
MLFLOW_TRACKING_URI = "http://localhost:5000" 

RANDOM_SEED = 42