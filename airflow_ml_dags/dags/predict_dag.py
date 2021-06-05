from datetime import timedelta, datetime

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator


default_args = {
    "owner": "anya_go",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "email_on_failure": False
}

DATA_VOLUMES = "/Users/anyya/Documents/data_analysis/MADE20/prod_ml_git/airflow_ml_dags/data"


with DAG(
        dag_id="predict",
        schedule_interval="@daily",
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

    wait_for_model = FileSensor(
        task_id="wait-model",
        filepath="{{ var.value.MODEL }}/model.pkl",
        poke_interval=5,
        retries=2
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input-dir /data/raw/{{ ds }} --model-dir {{ var.value.MODEL }} \
                 --output-dir /data/predictions/{{ ds }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=[f"{DATA_VOLUMES}:/data"]
    )

    [wait_for_raw_data, wait_for_target_data, wait_for_model] >> predict
