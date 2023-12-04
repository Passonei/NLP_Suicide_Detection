from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from feature_connector import FeatureConnector
from preprocessing import Preprocessor
import json
import pickle

config = json.load(open('config/config.json', 'r'))
        
procedure = config['preprocessing']['procedure']
preprocessor = Preprocessor(procedure=procedure)

model_path = config['models']['classifier']
vectorizer_path = config['models']['vectorizer']
pipeline_path = config['models']['pipeline']

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

connector = FeatureConnector(vectorizer)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('features', connector),
    ('classifier', model)
])

with open(pipeline_path, 'wb') as f:
    pickle.dump(pipe, f)