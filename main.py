#import pyaudio
import argparse
import os
from model.gender.model_utils import create_model, process_prediction
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
    male_prob = model.predict(features)
    gender, confidence = process_prediction(male_prob)
    text = speech_to_text(audio_file)

    print(f"Predicted: {gender}, with confidence: {confidence:.2f}")
    print(f"Detected text: {text}")