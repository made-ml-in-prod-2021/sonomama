import os
import pickle
from typing import List, Optional
import time

import pandas as pd
import uvicorn
from uvicorn import Config
import contextlib
import threading
from fastapi import FastAPI, HTTPException
from sklearn.pipeline import Pipeline

from features.schemas import Item, Response
from features.validate import is_valid


model: Optional[Pipeline] = None
SLEEP_ON_START_TIME = 20
APP_LIFE_TIME = 60


def load_pipeline(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


def make_prediction(data: List[Item],
                    pipeline: Pipeline) -> List[Response]:
    data_df = pd.DataFrame([record.__dict__ for record in data])
    predictions = pipeline.predict(data_df)
    return [Response(target=p) for p in predictions]


app = FastAPI()


@app.get("/")
def main():
    return "entry point of the predictor"


@app.on_event("startup")
def load_model():
    model_path = os.getenv("PATH_TO_MODEL", default="model.pkl")
    global model
    try:
        model = load_pipeline(model_path)
    except FileNotFoundError():
        raise HTTPException(status_code=500, detail="Model file not found")


@app.get("/health")
def health() -> bool:
    return not (model is None)


@app.post("/predict", response_model=List[Response])
def predict(request: List[Item]):
    if not health():
        raise HTTPException(status_code=500, detail="Was not able to load the model")
    for item in request:
        if not is_valid(item):
            raise HTTPException(status_code=400, detail="Data format is invalid")
    return make_prediction(request, model)


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


def run_app():
    server = Server(config=Config(app, host="0.0.0.0", port=os.getenv("PORT", 8000)))
    time.sleep(SLEEP_ON_START_TIME)
    with server.run_in_thread():
        time.sleep(APP_LIFE_TIME)


if __name__ == "__main__":
    run_app()

