import pyaudio
import argparse
import os
import wave
import librosa
import numpy as np
from sys import byteorder
from array import array
from struct import pack

from model.model_utils import create_model
from data.preprocess import Feature_Extractor

from utils import *

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_file', type=str,
                        help='Audio file to be used')
    
    parser.add_argument('--cfg', type=str,
                        default='parameters/cfg.yaml')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_arguments()
    audio_file = args.audio_file
    cfg = load_cfg(args.cfg)

    model = create_model()
    model.load_weights(cfg['model_weights'])

    if not audio_file or not os.path.isfile(audio_file):
        print("Please talk")
        record_to_file(cfg['output'])

    features = Feature_Extractor(audio_file, mel=True).extract().reshape(1, -1)
    male_prob = model.predict(features)[0][0]
    female_prob = 1 - male_prob
    gender = "male" if male_prob > female_prob else "female"
    print("Result:", gender)
    print(f"Probabilities:  Male: {male_prob*100:.2f}%    Female: {female_prob*100:.2f}%")