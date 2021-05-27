from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Union


@dataclass()
class PreprocessingParams:
    scaler: str = field(default=None)
