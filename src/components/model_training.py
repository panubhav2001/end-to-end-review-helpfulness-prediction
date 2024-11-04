import os
import joblib
import mlflow
from google.cloud import storage
from src.components.utils import load_data_from_bigquery, train_and_evaluate_model, log_best_model, deploy_model_to_vertex_ai
from src.logger import logger
from src.exception import CustomException
from src.components.preprocess import preprocess_data
import datetime
import pytz

# Define IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Set Google Cloud Storage bucket details
bucket_name = "amazon-reviews-project"
project_id = "airy-box-431604-j9"
region = "us-central1"

def main():
    # Create a unique experiment name
    experiment_name = f"Amazon_Reviews_Helpfulness_{datetime.datetime.now(IST).strftime('%Y-%m-%d_%H-%M-%S')}"
    mlflow.set_experiment(experiment_name)
    
    query = """
    SELECT * FROM `airy-box-431604-j9.amazon_reviews.clean_data`
    """
    try:
        logger.info("Loading data from BigQuery...")
        df = load_data_from_bigquery(query)
        logger.info(f"Loaded {len(df)} rows.")

        logger.info("Preprocessing data...")
        X, y = preprocess_data(df)  # Assuming preprocess_data function exists and is imported
        
        # Train and evaluate models
        logger.info("Training and evaluating models...")
        best_model, best_model_name, best_accuracy = train_and_evaluate_model(X, y, df)

        # Log and register the best model
        logger.info("Logging and registering the best model...")
        bucket_name = "amazon-reviews-project"  # Replace with your bucket name
        
        model_gcs_path = log_best_model(best_model, best_model_name, best_accuracy, bucket_name)
        
        #Deploy the model to endpoint
        deploy_model_to_vertex_ai(project_id, region, bucket_name, best_model_name, model_gcs_path)
        
    except CustomException as e:
        logger.error(e)

if __name__ == "__main__":
    main()
