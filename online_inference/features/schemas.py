from pydantic import BaseModel


class Item(BaseModel):
    age: int = 60
    sex: int = 1
    cp: int = 2
    trestbps: int = 145
    chol: int = 201
    fbs: int = 0
    restecg: int = 1
    thalach: int = 161
    exang: int = 0
    oldpeak: float = 1.4
    slope: int = 2
    ca: int = 1
    thal: int = 3


class Response(BaseModel):
    target: int
