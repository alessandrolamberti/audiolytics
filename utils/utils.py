import speech_recognition as sr
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

def create_model(vector_length=128):
    model = Sequential()
    model.add(Dense(256, input_shape=(vector_length,)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
    return model

def process_prediction(prediction):
    int_to_label = {0: 'male', 1: 'female'}

    prob_array = np.array([prediction, 1 - prediction])[:, 0, 0]
    gender = int_to_label[np.argmax(prob_array)]
    confidence = max(prob_array)
    return gender, confidence


def speech_to_text(wav):
    r = sr.Recognizer()
    with sr.AudioFile(wav) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Unintelligible text."
    return text

    