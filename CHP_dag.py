from datetime import timedelta
from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from extract_file import fetch_housing_data
from transform_file import transform
from load_file import load
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2020, 11, 8),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    dag_id='CHP_dag',
    default_args=default_args,
    description="Fetch and store dataset in 'Data' file",
    schedule_interval=timedelta(days=1),
)

extract = PythonOperator(
    task_id='extract_data',
    python_callable=fetch_housing_data,
    dag=dag,
)

transform = PythonOperator(
    task_id='transform_data',
    python_callable=transform,
    dag=dag,
)

build_model = PythonOperator(
    task_id='model_building',
    python_callable=load,
    dag=dag,
)


extract>>transform>>build_model