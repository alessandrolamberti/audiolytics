import yaml
import speech_recognition as sr

def load_cfg(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'),
            Loader=yaml.FullLoader)
    return cfg

def speech_to_text(wav):
    r = sr.Recognizer()
    with sr.AudioFile(wav) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Unintelligible text."
    return text

    