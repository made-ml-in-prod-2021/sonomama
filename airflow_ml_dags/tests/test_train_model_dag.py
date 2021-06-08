import pytest


@pytest.fixture(scope="module")
def structure():
    return {
        "wait-raw-data": ["docker-airflow-preprocess"],
        "wait-target-data": ["docker-airflow-preprocess"],
        "docker-airflow-preprocess": ["docker-airflow-split"],
        "docker-airflow-split": ["docker-airflow-train"],
        "docker-airflow-train": ["docker-airflow-validate"],
    }


def test_can_load_train_dag(dag_bag):
    dag = dag_bag.get_dag(dag_id="train_model")
    assert len(dag.tasks) == 6


def test_train_dag_has_correct_structure(dag_bag, structure):
    dag = dag_bag.get_dag(dag_id="train_model")
    for task_id, downstream_list in structure.items():
        assert dag.has_task(task_id), f"No task with id {task_id}"
        task = dag.get_task(task_id)
        assert (task.downstream_task_ids == set(downstream_list),
                f"Task downstream {task.downstream_task_ids} differs from"
                f"expected one {set(downstream_list)}")
