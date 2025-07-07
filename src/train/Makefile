clear-mlflow:
	@echo "Clearing MLflow cache..."
	@rm -rf mlruns
	@rm -rf mlartifacts
	@echo "MLflow mlruns and mlartifacts cleared."

start-mlflow:
	@echo "Starting MLflow server..."
	mlflow server \
		--host ${MLFLOW_HOST_IP} \
		--port ${MLFLOW_PORT} \

train:
	@echo "Running training script..."
	python main.py

setup:
	make start-mlflow & \
	celery -A train_worker worker --loglevel=info --queues=train