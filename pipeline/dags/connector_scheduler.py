"""
Airflow DAG for scheduling connector runs (incremental and backfill).

Daily Schedule:
- Incremental runs for all connectors (morning UTC)
- Fetches latest data for each source

Manual Triggers:
- Weekly backfill tasks (can be triggered on-demand)
- Monthly full backfill for historical data
"""

from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.decorators import apply_defaults

# Import connectors
from connectors.acled.connector import ACLEDConnector
from connectors.news.connector import NewsRSSConnector
from connectors.nbs.connector import NBSConnector
from connectors.social.nairaland_connector import NairalandConnector
from connectors.social.twitter_connector import TwitterConnector


def run_connector(connector_class: type, connector_name: str, mode: str = "incremental") -> dict[str, Any]:
    """
    Run a connector instance and return statistics.
    
    Args:
        connector_class: The connector class to instantiate
        connector_name: Human-readable name for logging
        mode: "incremental" or "backfill"
    
    Returns:
        Dictionary with run statistics (records_fetched, errors, duration)
    """
    import time
    from structlog import get_logger
    
    log = get_logger()
    start_time = time.time()
    
    try:
        connector = connector_class()
        log.info(
            "connector_run_started",
            connector=connector_name,
            mode=mode,
        )
        
        # Run connector with specified mode
        result = connector.run(mode=mode)
        
        duration = time.time() - start_time
        log.info(
            "connector_run_completed",
            connector=connector_name,
            mode=mode,
            duration_seconds=duration,
        )
        
        return {
            "connector": connector_name,
            "mode": mode,
            "status": "success",
            "duration_seconds": duration,
        }
    except Exception as e:
        duration = time.time() - start_time
        log.error(
            "connector_run_failed",
            connector=connector_name,
            mode=mode,
            error=str(e),
            duration_seconds=duration,
        )
        raise


# Default DAG arguments
default_args = {
    "owner": "ariran-pipeline",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email": ["contact@ariran.ng"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Main daily DAG for incremental runs
dag = DAG(
    "connector_incremental_daily",
    default_args=default_args,
    description="Daily incremental data fetches from all crime data sources",
    schedule_interval="0 6 * * *",  # 6 AM UTC daily
    catchup=False,
    tags=["connectors", "incremental"],
)

# Create tasks for each connector (incremental mode, runs in parallel)
task_news = PythonOperator(
    task_id="run_news_rss_incremental",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": NewsRSSConnector,
        "connector_name": "news_rss",
        "mode": "incremental",
    },
    dag=dag,
)

task_acled = PythonOperator(
    task_id="run_acled_incremental",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": ACLEDConnector,
        "connector_name": "acled",
        "mode": "incremental",
    },
    dag=dag,
)

task_twitter = PythonOperator(
    task_id="run_twitter_incremental",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": TwitterConnector,
        "connector_name": "twitter",
        "mode": "incremental",
    },
    dag=dag,
)

task_nairaland = PythonOperator(
    task_id="run_nairaland_incremental",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": NairalandConnector,
        "connector_name": "nairaland",
        "mode": "incremental",
    },
    dag=dag,
)

task_nbs = PythonOperator(
    task_id="run_nbs_incremental",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": NBSConnector,
        "connector_name": "nbs",
        "mode": "incremental",
    },
    dag=dag,
)

# All tasks run in parallel (no dependencies)
# Airflow will execute them concurrently


# Backfill DAG (on-demand, for weekly/monthly historical data)
dag_backfill = DAG(
    "connector_backfill_manual",
    default_args=default_args,
    description="On-demand backfill for historical data (triggered manually or weekly)",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=["connectors", "backfill"],
)

# Backfill tasks (with longer retry time due to volume)
backfill_args = default_args.copy()
backfill_args["retries"] = 3
backfill_args["retry_delay"] = timedelta(minutes=15)

task_backfill_news = PythonOperator(
    task_id="backfill_news_rss",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": NewsRSSConnector,
        "connector_name": "news_rss_backfill",
        "mode": "backfill",
    },
    retries=backfill_args["retries"],
    retry_delay=backfill_args["retry_delay"],
    dag=dag_backfill,
)

task_backfill_acled = PythonOperator(
    task_id="backfill_acled",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": ACLEDConnector,
        "connector_name": "acled_backfill",
        "mode": "backfill",
    },
    retries=backfill_args["retries"],
    retry_delay=backfill_args["retry_delay"],
    dag=dag_backfill,
)

task_backfill_twitter = PythonOperator(
    task_id="backfill_twitter",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": TwitterConnector,
        "connector_name": "twitter_backfill",
        "mode": "backfill",
    },
    retries=backfill_args["retries"],
    retry_delay=backfill_args["retry_delay"],
    dag=dag_backfill,
)

task_backfill_nairaland = PythonOperator(
    task_id="backfill_nairaland",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": NairalandConnector,
        "connector_name": "nairaland_backfill",
        "mode": "backfill",
    },
    retries=backfill_args["retries"],
    retry_delay=backfill_args["retry_delay"],
    dag=dag_backfill,
)

task_backfill_nbs = PythonOperator(
    task_id="backfill_nbs",
    python_callable=run_connector,
    op_kwargs={
        "connector_class": NBSConnector,
        "connector_name": "nbs_backfill",
        "mode": "backfill",
    },
    retries=backfill_args["retries"],
    retry_delay=backfill_args["retry_delay"],
    dag=dag_backfill,
)

# Backfill tasks also run in parallel
