
import os
import argparse
import logging
import mlflow
import json
import pandas as pd
from pathlib import Path
from mlflow.tracking import MlflowClient

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='Name under which model will be registered')
    parser.add_argument('--model_path', type=str, help='Model directory')
    parser.add_argument("--model_info_output_path", type=str, help="Path to write model info JSON")
    args, _ = parser.parse_known_args()
    print(f'Arguments: {args}')

    return args

def main(args):
    '''Loads the best-trained model from the sweep job and registers it'''

    print(f"Registering model: {args.model_name}")
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Error: Model path '{args.model_path}' does not exist.")
        
    # Initialize MLflow client
    client = MlflowClient()  # ADDED: Proper client initialization

    #  Load the trained model from the provided path
    model = mlflow.sklearn.load_model(args.model_path)
    
    #  Use actual artifact path from training
    artifact_path = "random_forest_price_regressor"  # Defined as variable for consistency

    # Log model using mlflow
    mlflow.sklearn.log_model(
        sk_model=model,
        registered_model_name=args.model_name,  # Specify the name under which the model will be registered
        artifact_path=artifact_path  # Specify the path where the model artifacts will be stored
    )

    # Register logged model using mlflow
    run_id = mlflow.active_run().info.run_id
    model_uri = f'runs:/{run_id}/{artifact_path}'
    
    # Register model
    mlflow_model = mlflow.register_model(model_uri, args.model_name)
    model_version = mlflow_model.version

    # Create output path
    output_dir = "./outputs"  # Azure ML automatically creates this writable directory
    os.makedirs(output_dir, exist_ok=True)  # This is now safe 

    # Write model info
    print("Writing JSON")
    model_info = {
        "id": f"{args.model_name}:{model_version.version}",
        "uri": model_version.source
    }
    output_path = os.path.join(output_dir, "model_info.json")
    with open(output_path, "w") as of:
        json.dump(model_info, of)

if __name__ == "__main__":
    
    with mlflow.start_run():
    
    # Parse Arguments
    args = parse_args()
    
    print("\n".join([
            f"Model name: {args.model_name}",
            f"Model path: {args.model_path}",
            f"Model info output path: {args.model_info_output_path}"
    ]))

    main(args)
