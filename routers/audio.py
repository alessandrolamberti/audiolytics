from typing import Optional
from fastapi import APIRouter, File, HTTPException
from fastapi.param_functions import Query
from starlette.responses import Response, FileResponse
from starlette.requests import Request
from config import gender_classifier, SHOW_ALL, logger, ALLOWED_FILE_EXTENSIONS, DO_SENTIMENT_ANALYSIS
from utils import *
import soundfile as sf
from io import BytesIO

router = APIRouter()


@router.post("/analytics/")
async def upload_file(request: Request, file: bytes = File(..., description="Audio wav file to analyse"),
                      tasks: Optional[str] = Query("gender, text, sentiment", description="List of tasks to perform")):
    
    response = {'success': False}
    text = None

    content_type = request._form['file'].content_type
    if content_type not in ALLOWED_FILE_EXTENSIONS:
        response["message"] = "File must be one of {}".format(", ".join(ALLOWED_FILE_EXTENSIONS))
        raise HTTPException(
            status_code=400, detail=response)

    file_like = BytesIO(file)
    data, rate = sf.read(BytesIO(file))
    
    if "gender" in tasks:
        features = extract_features(data, rate)
        gender, confidence = gender_prediction(features)
        response['audio analysis'] = {'gender': gender, 'confidence': confidence}

    if "text" in tasks:
        text, less_probable_text, text_confidence = speech_to_text(file_like, show_all=SHOW_ALL)
        response['text analysis'] = {'transcript': text, 'confidence': text_confidence, 'less_probable_transcripts': less_probable_text}

    if "sentiment" in tasks and DO_SENTIMENT_ANALYSIS:
        if text is None:
            text = speech_to_text(file_like, show_all=SHOW_ALL)[0]
        sentiment = text_sentiment(text)
        response['sentiment analysis'] = {'sentiment': sentiment}
    
    response['success'] = True

    return response


@router.post("/spectrogram/")
async def spectrogram(request: Request, file: bytes = File(..., description="Audio wav file to analyse")):
    
    content_type = request._form['file'].content_type
    if content_type not in ALLOWED_FILE_EXTENSIONS:
        response["message"] = "File must be one of {}".format(", ".join(ALLOWED_FILE_EXTENSIONS))
        raise HTTPException(
            status_code=400, detail=response)

    
    data, rate = sf.read(BytesIO(file))
    if len(data.shape) > 1:
        data = data[:, 0]

    spectrogram = create_spectrogram(data, rate)

    return Response(content=spectrogram.getvalue(), media_type='image/png')

