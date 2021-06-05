import os
from datetime import timedelta, datetime


from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "anya_go",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": "anyagoremykina@gmail.com"
}


with DAG(
        dag_id="download_data",
        schedule_interval="1 0 * * *",
        start_date=days_ago(3),
        end_date=datetime.today(),
        default_args=default_args
) as dag:

    download = DockerOperator(
        image="airflow-download",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-download",
        do_xcom_push=False,
        volumes=["/Users/anyya/Documents/data_analysis/MADE20/prod_ml_git/airflow_ml_dags/data:/data"]
    )
