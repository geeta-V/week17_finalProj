
import os
import argparse
import logging
import mlflow
import pandas as pd
from pathlib import Path

mlflow.start_run()  # Starting the MLflow experiment run

def main():
    # Argument parser setup for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the trained model")  # Path to the trained model artifact
    args = parser.parse_args()

    # Load the trained model from the provided path
    model = mlflow.sklearn.load_model(args.model)  # _ (Fill the code to load model from args.model)

    print("Registering the best trained used cars price prediction model")
    
    # Register the model in the MLflow Model Registry under the name "price_prediction_model"
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name="used_cars_price_prediction_model",  # Specify the name under which the model will be registered
        artifact_path="random_forest_price_regressor"  # Specify the path where the model artifacts will be stored
    )

    # End the MLflow run
    mlflow.end_run()  # __ (Fill in the code to end the MLflow run)

if __name__ == "__main__":
    main()
