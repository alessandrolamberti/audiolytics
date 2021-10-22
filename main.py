from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
import uvicorn
import os

from utils.preprocess import Feature_Extractor
from utils.utils import digest_features, speech_to_text

from config.get_cfg import model, SHOW_ALL, logger

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):

    response = {'success': False}
    
    if not file.filename.endswith(".wav"):
        return {"error": "File is not a wav file"}

    with open(file.filename, "wb") as f:
        f.write(file.file.read())
    
    features = Feature_Extractor(file.filename, mel=True).extract()
    gender, confidence = digest_features(features)
    response['audio prediction'] = {'gender': gender, 'confidence': confidence}

    text, less_probable_text, text_confidence = speech_to_text(file.filename, show_all=SHOW_ALL)
    response['text prediction'] = {'transcript': text, 'confidence': text_confidence, 'less_probable_transcripts': less_probable_text}

    response['success'] = True

    os.remove(file.filename)
    logger.info(f"File {file.filename} removed")

    return response

if __name__ == "__main__":
    uvicorn.run(app)
