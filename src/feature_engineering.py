import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureCreator():
    """
    FeatureCreator class to create features from text data.
    get_length() and get_exclamation_count() need to be run before preprocessing.
    get_sentiment() and get_keyword()need to be run after preprocessing.
    """
    def __init__(self, procedure):
        self.procedure = procedure
        self.vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
        self.vectorizer_trained = False
        self.sid = SentimentIntensityAnalyzer()
        if 'sentiment' in self.procedure:
            nltk.download('vader_lexicon')

    def get_length(self):
        return len(self.text.split())

    def get_exclamation_count(self):
        return len([word for word in self.text if word == '!'])

    def get_sentiment(self):
        sentiment = self.sid.polarity_scores(" ".join(self.text))
        if 'help' in self.text or 'pleas' in self.text and sentiment['compound']>0:
            sentiment['compound'] -= 0.3 
        return [value for value in sentiment.values()]
    
    def fit_vectorizer(self, corpus):
        self.vectorizer.fit_transform(corpus.apply(lambda x: ' '.join(x)))
        self.features = self.vectorizer.get_feature_names_out()
        self.vectorizer_trained = True
        return None

    def get_keyword(self):
        if not self.vectorizer_trained:
            raise Exception('Vectorizer not trained. Run fit_vectorizer() first.')
        else:
            vectorized_text = self.vectorizer.transform([' '.join(self.text)])
            keyword = self.features[vectorized_text.toarray().argsort()[0][::-1][0]]
            return keyword
    
    def fit(self, text):
        self.text = text
        feature_values = []
        for procedure in self.procedure:
            feature_values.append(getattr(self, procedure)())
        return feature_values