import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from starlette.responses import RedirectResponse

from src.text_summarizer.pipeline.prediction_pipeline import PredictionPipeline

text: str = "What is Text Summarization?"

app = FastAPI()


@app.get("/", tags=["authentication"])
async def index() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/train")
async def training() -> Response:
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predict_route(text: str) -> str:
    try:
        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
