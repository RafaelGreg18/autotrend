FROM python:3.10-slim

WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["celery", "-A", "celery_worker", "worker", "--loglevel=info"]