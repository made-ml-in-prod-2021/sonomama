import pytest
import json
from fastapi.testclient import TestClient
from app import app, load_model
from features.schemas import Item


client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def upload_model():
    load_model()


@pytest.fixture(scope="session")
def data_sample():
    return [Item(age=45, chol=99, oldpeak=1, trestbps=120, cp=0, restecg=0),
            Item()]


def test_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "entry point of the predictor"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json(), "model not found"


def test_predict(data_sample):
    response = client.post("/predict",
                           data=json.dumps([item.__dict__ for item in data_sample]))
    assert response.status_code == 200
    assert set([item["target"] for item in response.json()]).issubset([0, 1])


params_for_requests = [
    pytest.param(Item(ca=7), 400, id="Undefined feature category"),
    pytest.param(Item(age=500), 400, id="Numerical feature outside bounds")
]


@pytest.mark.parametrize(
    "data_sample, status_code",
    params_for_requests
)
def test_wrong_formats(data_sample, status_code):
    response = client.post("/predict",
                           data=json.dumps([data_sample.__dict__]))
    assert response.status_code == status_code











