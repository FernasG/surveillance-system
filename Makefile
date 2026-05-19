build:
	docker compose build

up:
	docker compose up

up-build:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f

sh\:guard:
	docker compose exec pi-guard bash

sh\:watcher:
	docker compose exec watcher bash