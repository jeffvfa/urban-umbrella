import pandas as pd 
import mlflow 
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from config.settings import RANDOM_SEED, PROCESSED_DATA_DIR


df = pd.read_parquet(PROCESSED_DATA_DIR) 

X = df[['sqft', 'bedrooms']]
y = df['price'] 

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