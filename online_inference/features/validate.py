from typing import Union
from features.schemas import Item

Number = Union[int, float]

CATEGORICAL = [("sex", 0, 1), ("cp", 0, 3), ("fbs", 0, 1),
               ("restecg", 0, 2), ("exang", 0, 1), ("slope", 0, 2),
               ("ca", 0, 4), ("thal", 0, 3)]

NUMERICAL = [("age", 1, 110), ("trestbps", 80, 250), ("chol", 80, 600),
             ("thalach", 50, 250), ("oldpeak", 0, 7)]


def validate_categorical(value: Number, lower: int, upper: int):
    if isinstance(value, float):
        raise ValueError("Expected integer for the categorical variable")
    elif value not in range(lower, upper + 1):
        raise ValueError("Undefined category")


def validate_numerical(value: Number, lower: Number, upper: Number):
    if not (lower <= value <= upper):
        raise ValueError(f"Numerical value outside expected {[lower, upper]} bounds")


def is_valid(item: Item) -> bool:
    try:
        for feature, lower, upper in CATEGORICAL:
            validate_categorical(getattr(item, feature), lower, upper)
        for feature, lower, upper in NUMERICAL:
            validate_numerical(getattr(item, feature), lower, upper)

        return True

    except ValueError:
        return False


