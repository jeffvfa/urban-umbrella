from feast import Entity, FileSource, FeatureView, Field 
from feast.types import Float32
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_DATA_PATH = BASE_DIR / "data/processed/features.parquet"


house_source = FileSource(path=str(PROCESSED_DATA_PATH), timestamp_field='event_timestamp')

house = Entity(name="house_id", join_keys=['house_id']) 

schema = [
    Field(name="sqft", dtype=Float32),
    Field(name="bedrooms", dtype=Float32),
]

house_features_view = FeatureView(name="house_features", entities=[house], ttl=None, schema=schema, source=house_source)