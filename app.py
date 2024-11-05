import streamlit as st
import pandas as pd
import logging
from src.pipeline.prediction_pipeline import load_resources, preprocess_and_predict

# Set up logging configuration
logging.basicConfig(
    filename="app_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load resources (vectorizers, transformers, model)
try:
    vectorizer_review, vectorizer_title, numerical_transformer, model = load_resources()
except Exception as e:
    logging.error(f"Error loading resources: {e}")
    st.error("Failed to load necessary resources.")
    st.stop()

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

        # Get prediction result
        result = preprocess_and_predict(input_data, vectorizer_review, vectorizer_title, numerical_transformer, model)

        # Display the result
        st.markdown(f"<h2 style='color: #6C757D;'>Predicted Helpfulness: <span style='color: #28A745;'>{result.capitalize()}</span></h2>", unsafe_allow_html=True)

    except Exception as e:
        logging.error(f"Error in prediction pipeline: {e}")
        st.error("Prediction error. Please try again.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
