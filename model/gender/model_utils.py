import pandas as pd 
import numpy as np
import os 
import tqdm 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

label_to_int = {"male": 1, "female": 0}

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
    #model.summary()
    return model

def load_data(vector_length = 128):
    if not os.path.isdir("../results"):
        os.mkdir("../results")
    if os.path.isfile("../results/features.npy") and os.path.isfile("../results/labels.npy"):
        X = np.load("../results/features.npy")
        y = np.load("../results/labels.npy")
        return X, y
    df = pd.read_csv("./data/balanced-all.csv")
    n_samples = len(df)
    n_male_samples = len(df[df['gender'] == 'male'])
    n_female_samples = len(df[df['gender'] == 'female'])
    print("Total samples:", n_samples)
    print("Total male samples:", n_male_samples)
    print("Total female samples:", n_female_samples)
    X = np.zeros((n_samples, vector_length))
    y = np.zeros((n_samples, 1))
    for i, (filename, gender) in tqdm.tqdm(enumerate(zip(df['filename'], df['gender'])), "Loading data", total=n_samples):
        features = np.load(f"./{filename}")
        X[i] = features
        y[i] = label_to_int[gender]
    np.save("../results/features", X)
    np.save("../results/labels", y)
    return X, y

def split(X, y, test_size=0.1, val_size=0.1):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=7)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=val_size, random_state=7)
    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_test": y_test
    }

def process_prediction(prediction):
    int_to_label = {0: 'male', 1: 'female'}

    prob_array = np.array([prediction, 1 - prediction])[:, 0, 0]
    gender = int_to_label[np.argmax(prob_array)]
    confidence = max(prob_array)
    return gender, confidence