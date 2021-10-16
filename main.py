import argparse
import os

from utils.preprocess import Feature_Extractor
from utils.utils import process_prediction, speech_to_text

from config.get_cfg import model

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_file', type=str,
                        help='Audio file to be used')
    
    return parser.parse_args()

if __name__ == '__main__':

    args = get_arguments()
    audio_file = args.audio_file

    features = Feature_Extractor(audio_file, mel=True).extract().reshape(1, -1)
    male_prob = model.predict(features)
    gender, confidence = process_prediction(male_prob)
    text = speech_to_text(audio_file)
    
    print("Gender predicted: {}".format(gender))
    print("Confidence: {:.3f}".format(confidence))
    print("Detected text: {}".format(text))