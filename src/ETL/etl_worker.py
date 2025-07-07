from celery import Celery
from main import get_ticker_data

app = Celery("etl_worker", broker="amqp://user:pass@rabbitmq:5672//")

@app.task
def run_etl_task(ticker_name: str, start_date: str | None):
    """
    Celery task to run the ETL process for a given ticker.
    
    Args:
        ticker_name (str): The name of the ticker to download data for.
        start_date (str | None): The start date for downloading data. If None, will download historical data from the beginning.
    """
    get_ticker_data(ticker_name, start_date)

def check_tast_status(task_id):
    """
    Check the status of a Celery task.
    
    Args:
        task_id (str): The ID of the Celery task to check.
            
    Returns:
        str: The status of the task.
    """
    result = app.AsyncResult(task_id)
    return result.status

if __name__ == "__main__":
    app.start(['worker', '--loglevel=info', '--queues=etl'])