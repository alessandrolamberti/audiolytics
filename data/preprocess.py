import pandas as pd 
import numpy as np 
import shutil 
import librosa 
import os
import glob
from tqdm import tqdm

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


if __name__ == '__main__':

    cvs_files = glob.glob("*.csv")

    for j, csv in enumerate(cvs_files):
        print("[INFO] Preprocessing", csv)
        df = pd.read_csv(csv)[["filename", "gender"]]
        df = df[np.logical_or(df['gender'] == 'female', df['gender'] == 'male')]

        new_csv = os.path.join("data", csv)
        df.to_csv(new_csv, index=False)

        folder_name, _ = csv.split(".")
        audio_files = glob.glob(f"{folder_name}/{folder_name}/*")
        all_audio_filenames = set(df["filename"])

        for i, audio_file in tqdm(list(enumerate(audio_files)), f"Extracting features of {folder_name}"):
                splited = os.path.split(audio_file)
                audio_filename = f"{os.path.split(splited[0])[-1]}/{splited[-1]}"

                if audio_filename in all_audio_filenames:
                    src_path = f"{folder_name}/{audio_filename}"
                    target_path = f"data/{audio_filename}"

                    if not os.path.isdir(os.path.dirname(target_path)):
                        os.mkdir(os.path.dirname(target_path))
                        
                    features = Feature_Extractor(src_path, mel=True).extract()
                    target_filename = target_path.split(".")[0]
                    np.save(target_filename, features)

