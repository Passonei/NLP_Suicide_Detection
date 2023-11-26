from preprocessing import Preprocessor
import json

if __name__ == '__main__':
    text = """I'm a very happy person :) :) :).
    Check out my website https://www.google.com/ and my twitter @twitter.
    In 2022 I will be 22 years old.
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
    # procedure = ['lower',
    #                 'remove_links',
    #                 'remove_punctuation',
    #                 'remove_numbers',
    #                 'translate_emoji',
    #                 'tokenize',
    #                 'remove_stopwords',
    #                 'lemmatize',
    #                 'remove_short_words'
    # ]

    preprocessor = Preprocessor(procedure=procedure)
    new_text = preprocessor.fit(text)
    
    print()
    print(" ".join(text.split()))
    print("-"*50)
    print(" ".join(new_text))