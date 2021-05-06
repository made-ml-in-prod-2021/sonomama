from ml_pipeline.data.make_dataset import split_train_test_data
from ml_pipeline.params import SplittingParams


TARGET_COL = "target"
DATASET_PATH = "raw_data/heart.csv"
FAKE_DATA_SIZE = 1000


def test_can_load_dataset(fake_data):
    assert len(fake_data) == 1000
    assert fake_data.isna().sum().sum() > 0, \
        "fake data does not have NaNs"
    assert TARGET_COL in fake_data.keys(), \
        "target columns is missing from the fake data"


def test_can_split_dataset(fake_data):
    test_size = 0.2
    splitting_params = SplittingParams(random_state=1, test_size=test_size,)
    train, test = split_train_test_data(fake_data, splitting_params)

    assert train.shape[0] == FAKE_DATA_SIZE * (1 - test_size), \
        "wrong train shape"
    assert test.shape[0] == FAKE_DATA_SIZE * test_size, \
        "wrong test shape"
