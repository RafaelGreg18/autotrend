from celery import Celery

app = Celery("producer", broker="amqp://user:pass@rabbitmq:5672//")

@app.task
def some_task(x, y):
    return x + y

def send_etl_task(ticker_name, start_date=None):
    app.send_task(
        "etl_worker.run_etl_task",
        args=[ticker_name, start_date]
    )