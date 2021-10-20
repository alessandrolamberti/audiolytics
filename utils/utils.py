import speech_recognition as sr
import numpy as np
import time
from config.get_cfg import logger, model

def process_prediction(prediction):
    int_to_label = {0: 'male', 1: 'female'}

    prob_array = np.array([prediction, 1 - prediction])[:, 0, 0]
    gender = int_to_label[np.argmax(prob_array)]
    confidence = float(max(prob_array))
    return gender, confidence

def digest_features(features):
    start = time.time()
    features.reshape(1, -1)
    gender, confidence = process_prediction(model.predict(features))
    end = time.time()
    logger.info("Feature digestion and prediction time: {}".format(end-start))

    return gender, confidence


def speech_to_text(wav, show_all=False):
    r = sr.Recognizer()
    text = ''
    confidence = None

    with sr.AudioFile(wav) as source:
        audio_data = r.record(source)
        start = time.time()
        try:
            if not show_all:
                text = r.recognize_google(audio_data, show_all=show_all)
            else:
                prediction = r.recognize_google(audio_data, show_all=show_all)['alternative']
                text = prediction[0]['transcript']
                confidence = prediction[0]['confidence']
        except sr.UnknownValueError:
            text = "Unintelligible text."
        end = time.time()
    logger.info("Time taken to recognize speech: {}".format(end - start))


    return text, confidence

    