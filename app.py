import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import aiplatform
from src.components.feature_engineering import add_feature_columns
import joblib
import logging

# Set up logging configuration
logging.basicConfig(
    filename="app_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize the Vertex AI client
aiplatform.init(project="airy-box-431604-j9", location="us-central1")

# Load the vectorizers (for preprocessing)
@st.cache_resource
def load_vectorizers():
    try:
        vectorizer_review = joblib.load("artifacts/tfidf_vectorizer_review.pkl")
        vectorizer_title = joblib.load("artifacts/tfidf_vectorizer_title.pkl")
        return vectorizer_review, vectorizer_title
    except Exception as e:
        logging.error(f"Error loading vectorizers: {e}")
        st.error("Failed to load vectorizers.")
        st.stop()

vectorizer_review, vectorizer_title = load_vectorizers()

# App Title with Description
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Amazon Review Helpfulness Predictor</h1>", unsafe_allow_html=True)
st.write("This app predicts the helpfulness of Amazon reviews based on the review text, title, and rating. Just fill in the review details, and the model will predict the level of helpfulness!")

# Input Form
st.header("Enter Review Details")
st.markdown("---")
title = st.text_input("Review Title")
review_text = st.text_area("Review Text")
rating = st.selectbox("Rating (out of 5)", options=[1, 2, 3, 4, 5])
submit_button = st.button(label="Predict Helpfulness")

# Prediction
if submit_button:
    try:
        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            "title": [title],
            "review_text": [review_text],
            "rating": [rating],
            "helpful_votes": [0]  # Placeholder for preprocessing consistency
        })

        # Preprocess the input data
        data_feature_engineered = add_feature_columns(input_data)
        data_feature_engineered = data_feature_engineered.drop(columns='helpful_votes', errors='ignore')

        # Preprocess numerical features
        numerical_cols = data_feature_engineered.select_dtypes(exclude=['object']).columns.tolist()
        numerical_transformer = joblib.load("artifacts/numerical_transformer.pkl")
        X_num = numerical_transformer.fit_transform(data_feature_engineered[numerical_cols])

        # Transform text features
        X_review_text = vectorizer_review.transform(input_data['review_text']).toarray()
        X_title = vectorizer_title.transform(input_data['title']).toarray()

        # Combine the numerical and text features
        X_transformed = np.hstack([X_num, X_review_text, X_title])

        # Convert features into a list for the API call
        instances = X_transformed.tolist()

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        st.error("Error in preprocessing. Please ensure all fields are correctly filled.")
        st.stop()

    try:
        # Specify the endpoint
        endpoint = aiplatform.Endpoint(endpoint_name="projects/airy-box-431604-j9/locations/us-central1/endpoints/1333400840748138496")

        # Make prediction request
        response = endpoint.predict(instances=instances)
        
        # Extract prediction result
        helpfulness_class = ["low", "medium", "high"]
        prediction = response.predictions[0]
        result = helpfulness_class[prediction]

        # Display the result
        st.markdown(f"<h2 style='color: #6C757D;'>Predicted Helpfulness: <span style='color: #28A745;'>{result.capitalize()}</span></h2>", unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        st.error("Prediction error. Please try again.")
        st.stop()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
