from fastapi import FastAPI, Depends
from fastapi.concurrency import run_in_threadpool
from prometheus_fastapi_instrumentator import Instrumentator
import logging
from typing import Callable
from pydantic import BaseModel
from time import time
from eval_onnx import predict as onnx_predict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    input: str



app = FastAPI()
Instrumentator().instrument(app).expose(app)

@app.get("/")
def root():
    return {
        'foo': 'bar'
    }


@app.get("/health-check")
def healthcheck():
    return {
        'status': 'OK'
    }


def get_model_predict():
    # Return the model prediction function
    return onnx_predict


@app.post("/api/predict")
async def predict(
    request: PredictRequest, model_predict: Callable = Depends(get_model_predict)
):
    t0 = time()

    prediction = await run_in_threadpool(model_predict, request.input)


    time_taken = (time() - t0) * 1000
    logger.info(f"{time_taken:.2f} ms spent on '{request.input}': {prediction}")

    return prediction


@app.post("/")
async def predict(
    request: PredictRequest, model_predict: Callable = Depends(get_model_predict)
):
    t0 = time()

    prediction = await run_in_threadpool(model_predict, request.input)


    time_taken = (time() - t0) * 1000
    logger.info(f"{time_taken:.2f} ms spent on '{request.input}': {prediction}")

    return prediction
