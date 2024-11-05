import pandas as pd
from datetime import datetime
import pytz
from google.cloud import bigquery, storage, aiplatform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import mlflow
import mlflow.sklearn
from src.logger import logger
from src.exception import CustomException
from mlflow.exceptions import MlflowException

# Initialize BigQuery client
client = bigquery.Client()
IST = pytz.timezone('Asia/Kolkata')  # Define IST timezone

def load_data_from_bigquery(query):
    """Load data from BigQuery."""
    try:
        df = client.query(query).to_dataframe()
        logger.info("Data loaded from BigQuery successfully.")
        return df
    except Exception as e:
        raise CustomException("Error loading data from BigQuery", e)

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model and return key metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)

        logger.info("Model evaluation completed.")
        return accuracy, precision, recall, f1, report
    except Exception as e:
        raise CustomException("Error evaluating the model", e)

def train_and_evaluate_model(X, y, dataset_df):
    """Train and evaluate multiple models, logging each run to MLflow with hyperparameters and dataset."""
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=10, random_state=42),
    }

    best_model = None
    best_f1_score = 0
    best_model_name = ""

    # Log dataset in MLflow
    timestamp = datetime.now(IST).strftime('%Y%m%d_%H%M%S')
    dataset_path = f"artifacts/datasets/dataset_{timestamp}.csv"
    dataset_df.to_csv(dataset_path, index=False)

    with mlflow.start_run(run_name="Dataset_Log") as dataset_run:
        mlflow.log_artifact(dataset_path, artifact_path="datasets")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_artifact(dataset_path, artifact_path="datasets")

            if model_name == "RandomForest":
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("max_depth", model.max_depth)
            elif model_name == "LogisticRegression":
                mlflow.log_param("max_iter", model.max_iter)
            elif model_name == "XGBoost":
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("max_depth", model.max_depth)

            model.fit(X_train, y_train)

            accuracy, precision, recall, f1, report = evaluate_model(model, X_test, y_test)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_text(report, "classification_report.txt")
            mlflow.sklearn.log_model(model, "model")

            if f1 > best_f1_score:
                best_f1_score = f1
                best_model = model
                best_model_name = model_name

            logger.info(f"{model_name} logged with f1 score: {f1:.4f}")

    return best_model, best_model_name, best_f1_score

def get_existing_model_performance(model_name):
    """Retrieve the F1 score of the existing model in the MLflow registry."""
    try:
        # Get the latest version of the registered model
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name)

        if latest_versions:
            latest_version = latest_versions[-1]
            run_id = latest_version.run_id

            # Get metrics from the run
            run_metrics = client.get_run(run_id).data.metrics
            existing_f1_score = run_metrics.get("f1_score", 0)
            return existing_f1_score
        else:
            logger.info(f"No registered versions of {model_name} found in the registry.")
            return None  # No registered model found

    except MlflowException as e:
        # Handle the specific error for model not found
        if "Registered Model with name" in str(e):
            logger.info(f"No registered model with name '{model_name}' found. Proceeding with new model registration.")
            return None  # Model not found, proceed with new registration
        else:
            raise CustomException(f"Error retrieving existing model performance for {model_name}", e)


def log_best_model(best_model, best_model_name, best_f1_score, bucket_name):
    """Log the best model to MLflow, register it, and upload it to Google Cloud Storage."""
    existing_f1_score = get_existing_model_performance(best_model_name)
    
    if existing_f1_score is None:
        logger.info("No existing model found. Proceeding with deployment of the new model.")
    else:
        logger.info(f"Existing model F1 score: {existing_f1_score:.4f}")
    
    # Proceed with model registration if no model exists or if the new model performs better
    if existing_f1_score is None or best_f1_score > existing_f1_score:
        logger.info(f"New model performs better or no existing model. Proceeding with deployment.")
        
        with mlflow.start_run(run_name="Best_Model_Deployment") as best_run:
            mlflow.log_param("model_name", best_model_name)
            mlflow.log_metric("f1_score", best_f1_score)
            mlflow.sklearn.log_model(best_model, "best_model")

            model_uri = f"runs:/{best_run.info.run_id}/best_model"
            mlflow.register_model(model_uri, best_model_name)
            logger.info(f"Best model {best_model_name} registered in MLflow Model Registry.")

            local_model_path = 'artifacts/best_model.pkl'
            joblib.dump(best_model, local_model_path)
            logger.info(f"Best model saved locally at {local_model_path}")

            mlflow.log_artifact(local_model_path, artifact_path="artifacts")
            logger.info("Best model and metrics logged to MLflow successfully.")
            
            # Upload model.pkl to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            gcs_model_path = f"model/{best_model_name}/model.pkl"
            blob = bucket.blob(gcs_model_path)
            blob.upload_from_filename(local_model_path)

            logger.info(f"Best model saved to GCS at gs://{bucket_name}/{gcs_model_path}")
            
            # Return the GCS directory (without the file name) for Vertex AI deployment
            return f"gs://{bucket_name}/model/{best_model_name}/"
    else:
        logger.info(f"Existing model performs better or equal to the new model. Skipping deployment.")


def deploy_model_to_vertex_ai(project_id, region, bucket_name, model_name, model_gcs_path):
    """Deploy the registered MLflow model to Google Vertex AI."""
    aiplatform.init(project=project_id, location=region)

    # Specify the container to use for deployment
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_gcs_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
    )

    endpoint = model.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=3,
    )

    logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
    print("Model deployed to endpoint:", endpoint.resource_name)
