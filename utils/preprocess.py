import librosa 
import numpy as np

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

    def __init__(self, file_name, **kwargs):
        self.file_name = file_name 
        self.mfcc = kwargs.get('mfcc')
        self.chroma = kwargs.get('chroma')
        self.mel = kwargs.get('mel')
        self.contrast = kwargs.get('contrast')
        self.tonnetz = kwargs.get('tonnetz')
    
    def extract(self):
        X, sample_rate = librosa.core.load(self.file_name)
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
