# Voice analysis endpoint

FastAPI based endpoint which loads a .wav file as input and returns a response.

## Example response
```
{
  "success": true,
  "audio prediction": {
    "gender": "male",
    "confidence": 0.9590235948562622
  },
  "text prediction": {
    "transcript": [
      "no that's not gonna do it"
    ],
    "confidence": [
      0.92713708
    ],
    "less_probable_transcripts": [
      "no that's not going to do it",
      "no that's not gonna do",
      "no that's not going to do",
      "now that's not gonna do it"
    ]
  }
}
```
## License

[MIT](https://choosealicense.com/licenses/mit/)

  
## Installation

Install instructions
```bash
  git clone https://github.com/alessandrolamberti/voice_analysis
  cd voice_analysis
  pip install -r requirements.txt
```





    
## Run Locally

Open API endpoint
```bash
uvicorn app:app
```

  
## Roadmap

- Add text sentiment analysis

- Customize API request

- Add audio sentiment analysis

- Improve temporary file storage (mySql?)