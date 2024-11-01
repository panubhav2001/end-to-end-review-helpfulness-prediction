import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from src.logger import logger
from src.exception import CustomException

# Download NLTK resources if not already available
nltk.download('punkt')

def flesch_kincaid(text):
    try:
        words = word_tokenize(text)
        sentences = len(re.split(r'[.!?]', text))
        syllables = sum([len([s for s in word if s in 'aeiou']) for word in words])
        if len(words) == 0 or sentences == 0:
            return np.nan
        return 206.835 - (1.015 * (len(words) / sentences)) - (84.6 * (syllables / len(words)))
    except Exception as e:
        raise CustomException("Error in flesch_kincaid calculation", e)

def pos_counts(text):
    try:
        words = word_tokenize(text)
        pos_tags = nltk.pos_tag(words)
        pos_counts = {"nouns": 0, "verbs": 0, "adjectives": 0}
        for _, tag in pos_tags:
            if tag.startswith('N'):
                pos_counts["nouns"] += 1
            elif tag.startswith('V'):
                pos_counts["verbs"] += 1
            elif tag.startswith('J'):
                pos_counts["adjectives"] += 1
        return pd.Series(pos_counts)
    except Exception as e:
        raise CustomException("Error calculating POS counts", e)

def add_feature_columns(df):
    """Adds various feature columns to the DataFrame based on text analysis."""
    try:
        logger.info("Starting feature engineering on data.")
        
        df['review_length'] = df['review_text'].apply(len)
        df['review_word_count'] = df['review_text'].apply(lambda x: len(word_tokenize(x)))
        df['review_sentiment'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['review_subjectivity'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        df['flesch_kincaid'] = df['review_text'].apply(flesch_kincaid)
        
        # Add keyword-based features
        keywords = ['good', 'bad', 'recommend', 'disappoint', 'excellent']
        for keyword in keywords:
            df[f'keyword_{keyword}'] = df['review_text'].apply(lambda x: int(keyword in x.lower()))
        
        # Calculate rating deviation
        avg_rating = df['rating'].mean()
        df['rating_deviation'] = df['rating'] - avg_rating
        df['title_sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['title_length'] = df['title'].apply(len)
        
        # Part of speech counts
        pos_features = df['review_text'].apply(pos_counts)
        df = pd.concat([df, pos_features], axis=1)
        
        # Negation and pronoun counts
        negations = ["not", "no", "never", "none"]
        df['negation_count'] = df['review_text'].apply(lambda x: sum([x.lower().count(neg) for neg in negations]))
        
        pronouns = ["i", "we", "you", "he", "she", "they"]
        df['pronoun_count'] = df['review_text'].apply(lambda x: sum([x.lower().count(pronoun) for pronoun in pronouns]))
        df['helpful_to_length_ratio'] = df['helpful_votes'] / (df['review_length'] + 1)

        logger.info("Feature engineering completed.")
        return df
    except Exception as e:
        raise CustomException("Error in add_feature_columns function", e)
