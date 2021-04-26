from dataclasses import dataclass


@dataclass()
class SplittingParams:
    test_size: float = .3
    random_state: int = 1984
