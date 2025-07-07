from celery import Celery

from main import train

app = Celery("train_worker", broker="amqp://user:pass@rabbitmq:5672//")

@app.task
def run_train_task(**kwargs):
    """
    Celery task to run the training process.
    This function will be called by the Celery worker.
    """
    print("Starting training task with parameters:", kwargs)
    train(**kwargs)
    print("Training task completed.")

if __name__ == "__main__":
    app.start(['worker', '--loglevel=info', '--queues=train'])