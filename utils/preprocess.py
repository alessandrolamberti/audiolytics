import librosa 
import numpy as np
import time
from config import logger

class Feature_Extractor():
    """
    This class extracts features from an audio file 'file_name'
    Supported:
     - MFCC: mfcc 
     - Chroma: chroma
     - MEL spectrogram frequency: mel
     - Contrast: contrast
     - Tonnetz: tonnetz
    """

    def __init__(self, data, rate, **kwargs):
        self.data = data
        self.rate = rate
        self.mfcc = kwargs.get('mfcc')
        self.chroma = kwargs.get('chroma')
        self.mel = kwargs.get('mel')
        self.contrast = kwargs.get('contrast')
        self.tonnetz = kwargs.get('tonnetz')
    
    def extract(self):
        X, sample_rate = self.data, self.rate
        if self.chroma or self.contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if self.mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if self.chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if self.mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if self.contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if self.tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
        return result


def extract_features(data, rate, mel=True):
    start = time.time()
    features = Feature_Extractor(data, rate, mel=mel).extract().reshape(1, -1)
    end = time.time()
    logger.info("Extracting features took {} seconds".format(end-start))
    return features
    
