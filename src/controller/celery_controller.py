from celery import Celery

app = Celery("producer", broker="amqp://user:pass@rabbitmq:5672//")

def send_etl_task(ticker_name, start_date=None):
    """Send ETL task to the etl queue"""
    result = app.send_task(
        "etl_worker.run_etl_task",
        args=[ticker_name, start_date],
        queue='etl'  # Route to etl queue
    )
    return result.id

def send_train_task(**kwargs):
    """
    Send a task to the Celery worker to run the training process.
    
    Args:
        **kwargs: Keyword arguments for the training process.
    """
    result = app.send_task(
        "train_worker.run_train_task",
        kwargs=kwargs,
        queue='train'  # Route to train queue
    )
    return result.id