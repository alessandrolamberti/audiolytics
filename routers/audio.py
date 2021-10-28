from typing import Optional
from fastapi import APIRouter, UploadFile, File
from fastapi.param_functions import Query
from starlette.responses import Response, FileResponse
from config.get_cfg import gender_classifier, SHOW_ALL, logger
from PIL import Image
from utils import *
import os
import io

router = APIRouter()


@router.post("/analytics/")
async def upload_file(file: UploadFile = File(..., description="Audio wav file to analyse"),
                      tasks: Optional[str] = Query("gender, text, sentiment", description="List of tasks to perform")):

    response = {'success': False}
    text = None
    
    if not file.filename.endswith(".wav"):
        return {"error": "File is not a wav file"}

    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    
    if "gender" in tasks:
        gender_features = Feature_Extractor(file.filename, mel=True).extract()
        gender, confidence = digest_features(gender_features)
        response['audio analysis'] = {'gender': gender, 'confidence': confidence}

    if "text" in tasks:
        text, less_probable_text, text_confidence = speech_to_text(file.filename, show_all=SHOW_ALL)
        response['text analysis'] = {'transcript': text, 'confidence': text_confidence, 'less_probable_transcripts': less_probable_text}

    if "sentiment" in tasks:
        if text is None:
            text = speech_to_text(file.filename, show_all=SHOW_ALL)[0]
        sentiment = text_sentiment(text)
        response['sentiment analysis'] = {'sentiment': sentiment}
    
    response['success'] = True

    os.remove(file.filename)
    logger.info(f"File {file.filename} removed")

    return response


@router.post("/spectrogram/")
async def spectrogram(file: UploadFile = File(..., description="Audio wav file to analyse")):
    
    if not file.filename.endswith(".wav"):
        return {"error": "File is not a wav file"}

    with open(file.filename, "wb") as f:
        f.write(file.file.read())


    create_spectrogram(file.filename)
    image = Image.open("spec.png")
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')

    os.remove(file.filename)
    os.remove("spec.png")
    

    return Response(content=image_bytes.getvalue(), media_type='image/png')

