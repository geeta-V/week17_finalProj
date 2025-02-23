
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
    parser.add_argument("--model_name", type=str, help="Name of the model to register")  # Model name for registration
    parser.add_argument("--model_path", type=str, help="Path to the trained model")  # Path to the trained model artifact
    parser.add_argument("--model_info_output_path", type=str, help="Path to save model info")  # Path to save model info
    args = parser.parse_args()

    # Load the trained model from the provided path
    model = mlflow.sklearn.load_model(args.model_path)  # _ (Fill the code to load model from args.model)

    print("Registering the best trained used cars price prediction model")
    print(f"Registering model: {args.model_name}")
    
    # Register the model in the MLflow Model Registry under the name "price_prediction_model"
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.model_name,  # Specify the name under which the model will be registered
        artifact_path="random_forest_price_regressor"  # Specify the path where the model artifacts will be stored
    )

     # Optionally, you could log some model information to a file
    model_info = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "status": "success"
    }
    # Save model info to the provided output path
    model_info_df = pd.DataFrame([model_info])
    model_info_df.to_csv(args.model_info_output_path, index=False)

    # End the MLflow run
    mlflow.end_run()  # __ (Fill in the code to end the MLflow run)

if __name__ == "__main__":
    main()
