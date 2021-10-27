import speech_recognition as sr
import numpy as np
import time
from config.get_cfg import logger, gender_classifier, BAD_RESPONSE
from transformers import pipeline

def process_prediction(prediction):
    int_to_label = {0: 'male', 1: 'female'}

    prob_array = np.array([prediction, 1 - prediction])[:, 0, 0]
    gender = int_to_label[np.argmax(prob_array)]
    confidence = float(max(prob_array))
    return gender, confidence

def digest_features(features):
    start = time.time()
    gender, confidence = process_prediction(gender_classifier.predict(features.reshape(1, -1)))
    end = time.time()
    logger.info("Feature digestion and gender prediction time: {}".format(end-start))

    return gender, confidence

def text_sentiment(text):
    if text == "Unintelligible text":
        return None
    start = time.time()
    sentiment = pipeline('sentiment-analysis')(text)[0]
    end = time.time()
    logger.info("Text sentiment prediction time: {}".format(end-start))

    return sentiment

def digest_audio_prediction(prediction, show_all):
    text = []
    confidence = []
    less_probable_text = []

    if len(prediction) == 0:
        return BAD_RESPONSE

    if not show_all:
        return text.append(prediction)
    prediction = prediction['alternative']
    text.append(prediction[0]['transcript'])
    confidence.append(prediction[0]['confidence'])
    for i in range(1,min(len(prediction), 2)):
        less_probable_text.append(prediction[i]['transcript'])
        if 'confidence' in prediction[i]:
            confidence.append(prediction[i]['confidence'])
    
    return text, less_probable_text, confidence

def speech_to_text(wav, show_all=True):
    r = sr.Recognizer()

    with sr.AudioFile(wav) as source:
        audio_data = r.record(source)
        try:
            start = time.time()
            prediction = r.recognize_google(audio_data, show_all=show_all)
            end = time.time()
            logger.info("Speech to text time: {}".format(end-start))

        except sr.UnknownValueError:
            logger.info("Google Speech Recognition could not understand audio")
            return BAD_RESPONSE

        except sr.RequestError as e:
            logger.info("Could not request results from Google Speech Recognition service; {}".format(e))
            return BAD_RESPONSE

    return digest_audio_prediction(prediction, show_all)
    