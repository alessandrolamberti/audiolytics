import speech_recognition as sr
import numpy as np
import time
import matplotlib.pyplot as plt
from config import logger, gender_classifier, BAD_RESPONSE
from transformers import pipeline
from io import BytesIO

def process_prediction(prediction):
    int_to_label = {0: 'male', 1: 'female'}

    prob_array = np.array([prediction, 1 - prediction])[:, 0, 0]
    gender = int_to_label[np.argmax(prob_array)]
    confidence = float(max(prob_array))
    return gender, confidence

def gender_prediction(features):
    start = time.time()
    gender, confidence = process_prediction(gender_classifier.predict(features))
    end = time.time()
    logger.info("Gender prediction time: {}".format(end-start))

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

    if 'confidence' in prediction[0].keys():
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

def create_spectrogram(data, rate):
    start = time.time()
    plt.figure(1)

    plot_a = plt.subplot(211)
    plot_a.plot(data)
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('Energy')

    plot_b = plt.subplot(212)
    plot_b.specgram(data, NFFT=1024, Fs=rate, noverlap=900)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    end = time.time()
    logger.info("Spectrogram creation time: {}".format(end-start))

    return buffer




    
    