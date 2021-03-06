import os
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

# Model
GENDER_MODEL_PATH = "./weights/model.h5"


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

ALLOWED_FILE_EXTENSIONS = ["audio/wav"]


# Speech recognition
SHOW_ALL = True # returns the most likely transcription if false, JSON complete response otherwise
BAD_RESPONSE = ("Unintelligible text", None, None)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DO_SENTIMENT_ANALYSIS = False

gender_classifier = create_model()
gender_classifier.load_weights(GENDER_MODEL_PATH)