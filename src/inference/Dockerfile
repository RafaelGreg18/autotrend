FROM python:3.11-slim

WORKDIR /inference

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 2000

CMD tail -f /dev/null