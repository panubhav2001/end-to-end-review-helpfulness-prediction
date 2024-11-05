import joblib
import numpy as np
import logging
from src.components.feature_engineering import add_feature_columns

# Load resources (vectorizers, transformers, model)
def load_resources():
    try:
        vectorizer_review = joblib.load("artifacts/tfidf_vectorizer_review.pkl")
        vectorizer_title = joblib.load("artifacts/tfidf_vectorizer_title.pkl")
        numerical_transformer = joblib.load("artifacts/numerical_transformer.pkl")
        model = joblib.load("artifacts/best_model.pkl")
        return vectorizer_review, vectorizer_title, numerical_transformer, model
    except Exception as e:
        logging.error(f"Error loading resources: {e}")
        raise e

# Preprocess and predict
def preprocess_and_predict(input_data, vectorizer_review, vectorizer_title, numerical_transformer, model):
    try:
        # Feature engineering
        data_feature_engineered = add_feature_columns(input_data)
        data_feature_engineered = data_feature_engineered.drop(columns='helpful_votes', errors='ignore')

        # Process numerical features
        numerical_cols = data_feature_engineered.select_dtypes(exclude=['object']).columns.tolist()
        X_num = numerical_transformer.transform(data_feature_engineered[numerical_cols])

        # Transform text features
        X_review_text = vectorizer_review.transform(input_data['review_text']).toarray()
        X_title = vectorizer_title.transform(input_data['title']).toarray()

        # Combine numerical and text features
        X_transformed = np.hstack([X_num, X_review_text, X_title])

        # Make prediction
        helpfulness_class = ["low", "medium", "high"]
        prediction = model.predict(X_transformed)
        result = helpfulness_class[prediction[0]]
        return result

    except Exception as e:
        logging.error(f"Error in preprocessing and prediction: {e}")
        raise e
