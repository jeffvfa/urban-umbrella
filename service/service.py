import bentoml 

from pydantic import BaseModel, Field

import pandas as pd 
 
from feast import feature_store 
from config.settings import FEATURE_STORE_PATH


#input schema 
class HouseRequest(BaseModel):
    house_id: int

# load model from bentoML
model = bentoml.sklearn.load_model("house_price_rf_model:latest")

# load Feast feature store 
store = feature_store.FeatureStore(repo_path=FEATURE_STORE_PATH) 

# Define the service using the @bentoml.service decorator
@bentoml.service(name="house_price")
class HousePriceService:        
    @bentoml.api
    def predict(self, input_data: HouseRequest) -> dict: 
        # create entity df for retrieval 
        house_id = input_data.house_id
        entity_row = [{
            "house_id": house_id
        }]
        
        # retrieve features from Feast
        features = [
            "house_features:sqft", 
            "house_features:bedrooms"
        ]
        
        features_from_store = store.get_online_features(features=features, entity_rows=entity_row)
        
        df = features_from_store.to_df()    
        
        X = df[["sqft", "bedrooms"]] 
        prediction = model.predict(X)
                
        return {"prediction" : prediction[0]}