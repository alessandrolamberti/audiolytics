import argparse
import os

from utils.preprocess import Feature_Extractor
from utils.utils import create_model, process_prediction, speech_to_text

from config.get_cfg import GENDER_MODEL_PATH

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio_file', type=str,
                        help='Audio file to be used')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    audio_file = args.audio_file

    model = create_model()
    model.load_weights(GENDER_MODEL_PATH)

    features = Feature_Extractor(audio_file, mel=True).extract().reshape(1, -1)
    male_prob = model.predict(features)
    gender, confidence = process_prediction(male_prob)
    text = speech_to_text(audio_file)

    print(f"Predicted: {gender}, with confidence: {confidence:.2f}")
    print(f"Detected text: {text}")