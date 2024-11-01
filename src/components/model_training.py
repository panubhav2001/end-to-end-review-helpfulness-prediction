import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pandas as pd
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.components.preprocess import preprocess_data  # Import the function
from sklearn.pipeline import Pipeline
from src.logger import logger
from src.exception import CustomException

# Initialize BigQuery client
client = bigquery.Client()

def load_data_from_bigquery(query):
    try:
        df = client.query(query).to_dataframe()
        logger.info("Data loaded from BigQuery successfully.")
        return df
    except Exception as e:
        raise CustomException("Error loading data from BigQuery", e)

def train_model(X_train, y_train):
    try:
        model =  RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return model
    except Exception as e:
        # Pass the error message as the first argument and the exception `e` as the second argument
        raise CustomException("Error training the model", e)

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logger.info("Model evaluation completed.")
        logger.info(f"Accuracy: {accuracy:.4f}\nClassification Report:\n{report}")
        return accuracy, report
    except Exception as e:
        raise CustomException("Error evaluating the model", e)

def save_model(model, model_path='artifacts/model.pkl'):
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        raise CustomException("Error saving the model", e)

def main():
    query = """
    SELECT * FROM `airy-box-431604-j9.amazon_reviews.clean_data`
    """
    try:
        logger.info("Loading data from BigQuery...")
        df = load_data_from_bigquery(query)
        logger.info(f"Loaded {len(df)} rows.")

        logger.info("Preprocessing data...")
        X, y = preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Training model...")
        model = train_model(X_train, y_train)
        
        logger.info("Evaluating model...")
        accuracy, report = evaluate_model(model, X_test, y_test)

        save_model(model)
    except CustomException as e:
        logger.error(e)

if __name__ == "__main__":
    main()
