from fastapi import FastAPI, UploadFile, File
import uvicorn
import os

from utils.preprocess import Feature_Extractor
from utils.utils import digest_features, speech_to_text

from config.get_cfg import model, SHOW_ALL, logger

app = FastAPI()


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    response = {'success': False}
    
    if not file.filename.endswith(".wav"):
        return {"error": "File is not a wav file"}

    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    
    response['filename'] = file.filename

    features = Feature_Extractor(file.filename, mel=True).extract()
    gender, confidence = digest_features(features)
    response['audio prediction'] = {'gender': gender, 'confidence': confidence}

    text, text_confidence = speech_to_text(file.filename, show_all=SHOW_ALL)
    response['text prediction'] = {'transcript': text, 'confidence': text_confidence}

    response['success'] = True

    os.remove(file.filename)
    logger.info(f"File {file.filename} removed")

    return response

if __name__ == "__main__":
    uvicorn.run(app)