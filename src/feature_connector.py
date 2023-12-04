import numpy as np
from scipy.sparse import hstack
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class FeatureConnector():
    """
    Class to connect the features extracted from the 
    text with the features extracted from the metadata.
    """
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        length = np.array(self.get_length(X)).reshape(-1,1)
        sentiments = np.array(self.get_sentiment(X)).reshape(-1,1)
        text_features = self.vectorizer.transform(X)
        return hstack([sentiments, length, text_features])

    def get_sentiment(self, texts):
        sentiments = []
        for text in texts:
            sentiment = SentimentIntensityAnalyzer().polarity_scores(text)
            help_words = [
            'help','suicide','plz', 'cyanide','ibuprofen','charcoal',
            'euthanasia','survivor','please','unimportant', 'insulin',
            'support', 'urgent', 'emergency']
            
            if any(word in (text) for word in help_words): 
                sentiment['compound'] *= 0.5
            sentiments.append(sentiment['compound'])
        return [sentiments]

    def get_length(self, texts):
        lengths = []
        for text in texts:
            lengths.append(len(text.split()))
        return [lengths]