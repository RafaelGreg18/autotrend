FROM python:3.11-slim

# Set the working directory
WORKDIR /orchestrator

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port for the orchestrator
EXPOSE 9000