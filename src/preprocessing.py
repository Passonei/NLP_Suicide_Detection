import nltk
import string
import re
import emoji
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords

class Preprocessor():
    """
    Preprocessor class to clean text data.
    """
    def __init__(self, procedure, max_length=200):
        nltk.download('punkt')
        nltk.download('vader_lexicon')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.procedure = procedure
        self.max_length = max_length
        self.translator = str.maketrans('', '', string.punctuation)
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.add('dont')
        self.stop_words.add('’')
        self.stop_words.add('\u200d')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def lower(self):
        return self.text.lower()

    def remove_punctuation(self):
        return self.text.translate(self.translator)           

    def remove_links(self):
        return re.sub(r'https?://\S+|www\.\S+|http\S+', '', self.text)

    def remove_numbers(self):
        return re.sub(r'\d+', '', self.text)

    def remove_emoji(self):
        return self.text.encode('ascii', 'ignore').decode('ascii')

    def translate_emoji(self):
        return emoji.demojize(self.text).replace(':', ' ')

    def tokenize(self):
        return word_tokenize(self.text)

    def remove_stopwords(self):
        return [word for word in self.text if word not in self.stop_words]

    def stem(self):
        return ([self.stemmer.stem(word) for word in self.text])

    def lemmatize(self):
        return [self.lemmatizer.lemmatize(word) for word in self.text]
    
    def remove_singular_marks(self):
        return [word for word in self.text if len(word) > 1]

    def remove_short_words(self):
        return [word for word in self.text if len(word) > 3]

    def remove_long_words(self):
        return [word for word in self.text if len(word) < self.max_length]

    def shorten_text(self):
        return self.text[:self.max_length]

    def transform(self, texts):
        preprocessed = []
        for text in texts:
            self.text = text
            for procedure in self.procedure:
                self.text = getattr(self, procedure)()
            preprocessed.append(' '.join(self.text))
        return preprocessed
    
    def fit(self, X, y=None):
        return self