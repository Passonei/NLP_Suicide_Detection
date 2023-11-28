from preprocessing import Preprocessor
from feature_engineering import FeatureCreator
import json

if __name__ == '__main__':
    text = """I'm a very happy person :) :) :).
    Check out my website https://www.google.com/ and my twitter @twitter.
    In 2022 I will be 22 years old!!!.
    I find weird post: 
    'pee probably tastes like salty tea😏💦‼️ 
    can someone who drank pee before confirm this🙄‼️'
    I'm not sure if I should remove the emoji or translate them.
    i didnt know that i was starving till i tasted you
    dont you wanna dance with me no more
    """
    with open('config/config.json', 'r') as file:
        config = json.load(file)
    procedure = config['preprocessing']['procedure']

    # feature engineering before preprocessing
    procedure_feature_preclean = config['feature_engineering']['procedure_before_preprocessing']

    feature_creator_preclean = FeatureCreator(procedure=procedure_feature_preclean)
    feature_preclean = feature_creator_preclean.fit(text)

    # preprocessing
    preprocessor = Preprocessor(procedure=procedure)
    new_text = preprocessor.fit(text)

    # feature engineering after preprocessing
    import pandas as pd
    procedure_feature_postclean = config['feature_engineering']['procedure_after_preprocessing']
    feature_creator_postclean = FeatureCreator(procedure=procedure_feature_postclean)

    df = pd.read_csv('data\prepared\prepared.csv', usecols=['corpus','class'])
    df['corpus'] = df['corpus'].apply(lambda x: x[1:-1].replace("'", "").split(', '))

    feature_creator_postclean.fit_vectorizer(df['corpus'])
    
    feature_post_clean = feature_creator_postclean.fit(new_text)
    
    print("\nOriginal text:")
    print(" ".join(text.split()))
    print("\n" + "-"*25 + " PREPROCESSING " + "-"*25)
    print(" ".join(new_text))
    print("\n" + "-"*25 + " FEATURE ENGINEERING " + "-"*25)
    print(f'Preclean features - Length: {feature_preclean[0]}, Exclamation count: {feature_preclean[1]}')
    print(f'Postclean features - Sentiment: {feature_post_clean[0]}, Keyword: {feature_post_clean[1]}')
    print("-"*50)