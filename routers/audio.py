from fastapi import APIRouter, UploadFile, File, Request
import os
from utils.preprocess import Feature_Extractor
from utils.utils import digest_features, speech_to_text

from config.get_cfg import gender_classifier, SHOW_ALL, logger


router = APIRouter()


@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    response = {'success': False}
    
    if not file.filename.endswith(".wav"):
        return {"error": "File is not a wav file"}

    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    
    gender_features = Feature_Extractor(file.filename, mel=True).extract()
    gender, confidence = digest_features(gender_features)
    response['audio analysis'] = {'gender': gender, 'confidence': confidence}

    text, less_probable_text, text_confidence = speech_to_text(file.filename, show_all=SHOW_ALL)
    response['text prediction'] = {'transcript': text, 'confidence': text_confidence, 'less_probable_transcripts': less_probable_text}

    response['success'] = True

    os.remove(file.filename)
    logger.info(f"File {file.filename} removed")

    return response
