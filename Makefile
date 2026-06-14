-include .env

export

.PHONY: test

build:
	docker compose build

up:
	docker compose up -d
	@echo "Checking if model $(OLLAMA_MODEL_NAME) is installed..."
	@docker compose exec -T ollama ollama list | grep -q "$(OLLAMA_MODEL_NAME)" && echo "Model $(OLLAMA_MODEL_NAME) is already installed!" || \
		(echo "Model not found. Starting download of $(OLLAMA_MODEL_NAME)..." && \
		 docker compose exec -T ollama ollama pull $(OLLAMA_MODEL_NAME) && \
		 echo "Download completed!")
	docker compose logs -f

up-build:
	docker compose up --build -d
	@echo "Checking if model $(OLLAMA_MODEL_NAME) is installed..."
	@docker compose exec -T ollama ollama list | grep -q "$(OLLAMA_MODEL_NAME)" && echo "Model $(OLLAMA_MODEL_NAME) is already installed!" || \
		(echo "Model not found. Starting download of $(OLLAMA_MODEL_NAME)..." && \
		 docker compose exec -T ollama ollama pull $(OLLAMA_MODEL_NAME) && \
		 echo "Download completed!")
	docker compose logs -f

down:
	docker compose down

logs:
	docker compose logs -f

sh\:guard:
	docker compose exec pi-guard bash

sh\:watcher:
	docker compose exec watcher bash

sh\:ml:
	docker compose exec ml-server bash