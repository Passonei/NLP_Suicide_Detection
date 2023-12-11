import pickle
import json

if __name__ == '__main__':
    text = """"No one ever picks up.Whenever I need help, people just post the Hotline number. Yet when I call, no one ever picks up. Is there no point then?\n\nI just want a hug and someone to tell me it's going to be okay, that I'm not a failure.\n\nI can't even commit suicide right."
    """
    config = json.load(open('config/config.json', 'r'))
    model_path = config['models']['pipeline']
    model = pickle.load(open(model_path, 'rb'))

    y = model.predict([text])
    proba = model.predict_proba([text])
    print("\n" + "-"*25 + " PREDICTION " + "-"*25)
    print(f'Prediction: {y[0]}, Probability {(proba[0][y[0]]*100).round(2)}%')