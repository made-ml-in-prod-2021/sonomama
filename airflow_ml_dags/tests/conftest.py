from airflow.models import DagBag
import pytest


@pytest.fixture(scope="session")
def dag_bag():
    return DagBag(dag_folder='dags/', include_examples=False)
