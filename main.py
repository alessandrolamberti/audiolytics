from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
import uvicorn
from routers import audio


app = FastAPI()

app.include_router(router=audio.router, prefix="/audio", tags=["audio"])

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down")

    

if __name__ == "__main__":
    uvicorn.run(app)
