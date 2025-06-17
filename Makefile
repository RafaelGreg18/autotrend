up:
	@mkdir data_lake data_warehouse model_registry && \
	docker compose up

down:
	docker compose down

clear-data:
	rm data_lake/*
	rm data_warehouse/*

clear-models:
	rm model_registry/*

clear-all:
	rm data_lake/*
	rm data_warehouse/*
	rm model_registry/*