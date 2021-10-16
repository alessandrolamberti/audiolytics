from fastapi import FastAPI, UploadFile, File
import uvicorn
import os

from utils.preprocess import Feature_Extractor
from utils.utils import process_prediction, speech_to_text

from config.get_cfg import model

def build_response(gender, confidence, text):
    response = {
        'gender': gender,
        'confidence': str(confidence), 
        'predicted text': text
    }

    return response

app = FastAPI()


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to the server.
    """
    #check if file is a wav file, http error otherwise
    if not file.filename.endswith(".wav"):
        return {"error": "File is not a wav file"}

    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    features = Feature_Extractor(file.filename, mel=True).extract().reshape(1, -1)
    male_prob = model.predict(features)
    gender, confidence = process_prediction(male_prob)
    text = speech_to_text(file.filename)

    os.remove(file.filename)

    return {"filename": file.filename, 
            "prediction": build_response(gender, confidence, text)}

if __name__ == "__main__":
    uvicorn.run(app)