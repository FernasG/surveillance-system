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

shell:
	docker compose exec app sh