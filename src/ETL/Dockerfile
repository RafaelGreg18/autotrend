FROM python:3.10-slim

WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

# This will be overridden by docker-compose, but good to have as default
CMD ["celery", "-A", "etl_worker", "worker", "--loglevel=info", "--queues=etl"]