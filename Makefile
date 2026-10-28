COMPOSE_DEV = docker compose -f docker-compose.yml -f docker-compose.dev.yml

build:
	docker compose build

up:
	docker compose up

up-build:
	docker compose up --build

dev:
	$(COMPOSE_DEV) up

dev-build:
	$(COMPOSE_DEV) up --build

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

ca-cert:
	docker compose cp caddy:/data/caddy/pki/authorities/local/root.crt caddy-root.crt
