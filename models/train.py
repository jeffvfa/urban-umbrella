import pandas as pd 
import mlflow 
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from feast import FeatureStore 

import bentoml

from config.settings import RANDOM_SEED, PROCESSED_DATA_DIR, FEATURE_STORE_PATH


store = FeatureStore(repo_path=FEATURE_STORE_PATH)

# load dataset
df = pd.read_parquet(PROCESSED_DATA_DIR) 

entity_df = df[["house_id", "event_timestamp"]].copy()
features = ["house_features:sqft", "house_features:bedrooms"]

features_from_store = store.get_historical_features(entity_df=entity_df, features=features)

df_features = features_from_store.to_df()

# normalize timestamps
df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
df_features["event_timestamp"] = pd.to_datetime(df_features["event_timestamp"], utc=True)


training_df = df_features.merge(
    df[["house_id", "event_timestamp", "price"]],
    on=["house_id", "event_timestamp"]
)


X = training_df[['sqft', 'bedrooms']]
y = training_df['price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

mlflow.set_experiment("house_price_prediction") 

with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=10, random_state=RANDOM_SEED) 
    model.fit(X_train, y_train) 

    predictions = model.predict(X_test) 
    mse = mean_squared_error(y_test, predictions) 

    mlflow.log_param("n_estimators", 10) 
    mlflow.log_metric("mse", mse) 
    mlflow.sklearn.log_model(model, name="random_forest_model") 

    print(f"Model training complete. MSE: {mse}") 
    
bentoml.sklearn.save_model("house_price_rf_model", model)