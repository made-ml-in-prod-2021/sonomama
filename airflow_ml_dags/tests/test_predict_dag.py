import pytest


@pytest.fixture(scope="module")
def structure():
    return {
        "wait-raw-data": ["docker-airflow-predict"],
        "wait-target-data": ["docker-airflow-predict"],
        "wait-model": ["docker-airflow-predict"],
    }


def test_can_load_predict_dag(dag_bag):
    dag = dag_bag.get_dag(dag_id="predict")
    assert len(dag.tasks) == 4


def test_predict_dag_has_correct_structure(dag_bag, structure):
    dag = dag_bag.get_dag(dag_id="predict")
    for task_id, downstream_list in structure.items():
        assert dag.has_task(task_id), f"No task with id {task_id}"
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)
