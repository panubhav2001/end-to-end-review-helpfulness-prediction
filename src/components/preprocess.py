import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from src.components.feature_engineering import add_feature_columns  # Importing feature engineering functions
from src.logger import logger
from src.exception import CustomException
import sys
import numpy as np
import joblib  # Importing joblib for saving models

class TfidfVectorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=100):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        return self.vectorizer.transform(X).toarray()

def preprocess_data(df):
    try:
        logger.info("Starting data preprocessing.")

        # Drop unnecessary columns and handle missing values
        df.drop(columns='review_hash', inplace=True, errors='ignore')
        df['helpful_votes'] = pd.to_numeric(df['helpful_votes'], errors='coerce').fillna(0).astype(int)
        df = df.dropna(subset=['review_text', 'title', 'rating', 'helpful_votes'])

        # Feature engineering and target creation
        df = add_feature_columns(df)
        bins = [0, 1, 5, float("inf")]
        labels = ["low", "medium", "high"]
        df['helpfulness_class'] = pd.cut(df['helpful_votes'], bins=bins, labels=labels)
        df['helpfulness_class'] = df['helpfulness_class'].fillna("low")
        label_encoder = LabelEncoder()
        df['helpfulness_class_encoded'] = label_encoder.fit_transform(df['helpfulness_class'])

        X = df.drop(columns=['helpful_votes', 'helpfulness_class', 'helpfulness_class_encoded'])
        y = df['helpfulness_class_encoded']

        if 'review_text' not in X.columns or 'title' not in X.columns:
            raise CustomException("Required text columns ('review_text', 'title') are missing after preprocessing", sys)

        # Numerical column transformation
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        X_num = numerical_transformer.fit_transform(X[numerical_cols])
        
        # Save the numerical transformer
        joblib.dump(numerical_transformer, 'artifacts/numerical_transformer.pkl')
        logger.info("Numerical Transformer saved as pickle file")
        
        # 'review_text' column transformation
        review_text_transformer = TfidfVectorizerTransformer(max_features=100)
        X_review_text = review_text_transformer.fit_transform(X['review_text'])

        # 'title' column transformation
        title_transformer = TfidfVectorizerTransformer(max_features=100)
        X_title = title_transformer.fit_transform(X['title'])
        
        logger.info("Saving Tfidf Vectorizers as pickle file")
        # Save the TF-IDF vectorizers
        joblib.dump(review_text_transformer.vectorizer, 'artifacts/tfidf_vectorizer_review.pkl')
        joblib.dump(title_transformer.vectorizer, 'artifacts/tfidf_vectorizer_title.pkl')
        logger.info("Saved Tfidf Vectorizers")
        # Concatenate all transformed parts
        X_transformed = np.hstack([X_num, X_review_text, X_title])

        logger.info("Data preprocessing completed successfully.")
        return X_transformed, y

    except Exception as e:
        raise CustomException("Error in preprocess_data function", e)
