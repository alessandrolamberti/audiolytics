import numpy as np
from sys import byteorder
from array import array
from struct import pack
import pyaudio
import wave
import yaml
import speech_recognition as sr

def load_cfg(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'),
            Loader=yaml.FullLoader)
    return cfg


cfg = load_cfg('parameters/cfg.yaml')
RATE = cfg['RATE']
CHUNK_SIZE = cfg['CHUNK_SIZE']
THRESHOLD = cfg['THRESHOLD']
SILENCE = cfg['SILENCE']



def is_silent(snd):
    return max(snd) < THRESHOLD

def normalize(snd):
    MAX = 16384
    times = float(MAX)/max(abs(i) for i in snd)

    r = array('h')
    for i in snd:
        r.append(int(i*times))
    return r

def trim(snd):
    "Trim the blank spots at the start and end"
    def _trim(snd):
        snd_started = False
        r = array('h')

        for i in snd:
            if not snd_started and abs(i)> THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r
    
    # Trim to the left
    snd = _trim(snd)

    # Trim to the right
    snd.reverse()
    snd = _trim(snd)
    snd.reverse()
    return snd

def add_silence(snd, seconds):
    "Add silence to the start and end of 'snd' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while True:
        # little endian, signed short
        snd = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd.byteswap()
        r.extend(snd)

        silent = is_silent(snd)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(pyaudio.paInt16)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def speech_to_text(wav):
    r = sr.Recognizer()
    with sr.AudioFile(wav) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
    return text

    