from datetime import timedelta, datetime

from airflow import DAG, models
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    "owner": "anya_go",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "email": models.Variable.get("email"),
    "email_on_failure": True,
    "email_on_retry": True,
}

DATA_VOLUMES = "/Users/anyya/Documents/data_analysis/MADE20/prod_ml_git/airflow_ml_dags/data"


with DAG(
        dag_id="train_model",
        schedule_interval="@weekly",
        start_date=datetime.today(),
        default_args=default_args
) as dag:

    wait_for_raw_data = FileSensor(
        task_id="wait-raw-data",
        filepath="data/raw/{{ ds }}/data.csv",
        poke_interval=5,
        retries=2
    )

    wait_for_target_data = FileSensor(
        task_id="wait-target-data",
        filepath="data/raw/{{ ds }}/target.csv",
        poke_interval=5,
        retries=2
    )

    preprocess = DockerOperator(
        task_id="docker-airflow-preprocess",
        image="airflow-preprocess",
        command="--input-dir /data/raw/{{ ds }} --output-dir /data/processed/{{ ds }}",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUMES}:/data"]
    )

    split = DockerOperator(
        task_id="docker-airflow-split",
        image="airflow-split",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/processed/{{ ds }}",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUMES}:/data"]
    )

    train_model = DockerOperator(
        task_id="docker-airflow-train",
        image="airflow-train-model",
        command="--input-dir /data/processed/{{ ds }} --output-dir /data/models/{{ ds }}",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUMES}:/data"]
    )

    validate = DockerOperator(
        task_id="docker-airflow-validate",
        image="airflow-validate",
        command="--input-dir /data/processed/{{ ds }} --model-dir /data/models/{{ ds }}",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUMES}:/data"]
    )

    [wait_for_raw_data, wait_for_target_data] >> preprocess >> split >> train_model >> validate
