def test_can_load_download_dag(dag_bag):
    dag = dag_bag.get_dag(dag_id="download_data")
    assert len(dag.tasks) == 1
    assert "docker-airflow-download" in dag.task_dict

