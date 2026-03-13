#!/bin/bash

kill -9 $(lsof -t -i:3000)
kill -9 $(lsof -t -i:5000)


set -e

source .venv/bin/activate

echo "Starting ML pipeline..."

# ---------------------------
# Start MLflow
# ---------------------------
if lsof -i :5000 > /dev/null
then
    echo "MLflow already running at http://127.0.0.1:5000"
else
    echo "Starting MLflow UI..."
    mlflow ui --host 127.0.0.1 --port 5000 > logs_mlflow.txt 2>&1 &
    MLFLOW_PID=$!
fi


echo "Step 1: Data processing"
python -m pipelines.data_pipeline


echo "Step 2: Update Feast feature registry"
(
cd feature_store/house_features/feature_repo
feast apply
)


echo "Step 3: Train model"
python -m models.train


# ---------------------------
# Start BentoML
# ---------------------------
if lsof -i :3000 > /dev/null
then
    echo "BentoML already running at http://127.0.0.1:3000"
else
    echo "Starting BentoML..."
    bentoml serve service/service.py:HousePriceService --reload > logs_bentoml.txt 2>&1 &
    BENTO_PID=$!
fi


echo "Servers running:"
echo "MLflow PID: $MLFLOW_PID"
echo "BentoML PID: $BENTO_PID"

echo "Pipeline completed successfully!"