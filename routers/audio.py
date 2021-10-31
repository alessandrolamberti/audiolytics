from typing import Optional
from fastapi import APIRouter, UploadFile, File
from fastapi.param_functions import Query
from starlette.responses import Response, FileResponse
from config import gender_classifier, SHOW_ALL, logger
from utils import *
import soundfile as sf
from io import BytesIO

router = APIRouter()


@router.post("/analytics/")
async def upload_file(file: bytes = File(..., description="Audio wav file to analyse"),
                      tasks: Optional[str] = Query("gender, text, sentiment", description="List of tasks to perform")):

    response = {'success': False}
    text = None
    file_like = BytesIO(file)
    data, rate = sf.read(BytesIO(file))
    
    if "gender" in tasks:
        features = extract_features(data, rate)
        gender, confidence = gender_prediction(features)
        response['audio analysis'] = {'gender': gender, 'confidence': confidence}

    if "text" in tasks:
        text, less_probable_text, text_confidence = speech_to_text(file_like, show_all=SHOW_ALL)
        response['text analysis'] = {'transcript': text, 'confidence': text_confidence, 'less_probable_transcripts': less_probable_text}

    if "sentiment" in tasks:
        if text is None:
            text = speech_to_text(file_like, show_all=SHOW_ALL)[0]
        sentiment = text_sentiment(text)
        response['sentiment analysis'] = {'sentiment': sentiment}
    
    response['success'] = True

    return response


@router.post("/spectrogram/")
async def spectrogram(file: bytes = File(..., description="Audio wav file to analyse")):
    
    data, rate = sf.read(BytesIO(file))
    spectrogram = create_spectrogram(data, rate)

    return Response(content=spectrogram.getvalue(), media_type='image/png')

