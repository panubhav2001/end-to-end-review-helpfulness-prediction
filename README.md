# Amazon Review Helpfulness Prediction App

This repository contains the **Streamlit front-end application** for predicting the helpfulness of Amazon reviews using a machine learning model. The>

## Table of Contents
- [App Overview](#app-overview)
- [Architecture Summary](#architecture-summary)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [App Usage](#app-usage)
- [Future Improvements](#future-improvements)

## App Overview

The application is a **Streamlit** front-end designed to perform two main functions:

1. **Training Pipeline**:
   - **Data Ingestion**: Reads cleaned review data from a BigQuery table.
   - **Data Processing**: Performs feature engineering, text preprocessing, and transformations.
   - **Model Training**: Trains a classification model to predict review helpfulness.
   - **Logging**: Logs all models to **MLflow**. If the model outperforms previous versions, it is saved to the Vertex AI model registry.

2. **Prediction Pipeline**:
   - Loads the saved **TF-IDF model**, **numerical transformers**, and the trained classification model.
   - Takes user input (a review) and returns a prediction on whether the review is likely helpful.

## Architecture Summary

The app is part of a larger architecture that performs the following steps:

1. **Data Collection**: A **Dataproc PySpark** job, triggered by a **Cloud Scheduler**, scrapes Amazon reviews daily and publishes messages to a **Pu>
2. **Data Storage**: The messages are streamed to **BigQuery** via a Pub/Sub subscription.
3. **Data Cleaning**: A scheduled query processes the raw data, cleans it, and writes it to a BigQuery table.
4. **Model Training & Prediction**: The **Streamlit app** (this repository) reads the cleaned data, trains the model, and logs it to MLflow. The app >

## Technologies Used
- **Google Cloud Platform**:
  - **BigQuery**: For storing and managing the cleaned review data.
  - **Pub/Sub**: For data streaming and message passing.
- **Mlflow**: For model registry and tracking.
- **Streamlit**: To provide an interactive front-end for training and predictions.
- **PySpark**: Used in the larger architecture for initial data scraping and processing.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/amazon-review-helpfulness-app.git
   cd amazon-review-helpfulness-app
