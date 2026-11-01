.PHONY: up down models migrate api worker test test-integration lint eval bakeoff hooks

up:
	docker compose up -d --wait

down:
	docker compose down

models:
	scripts/pull-models.sh

migrate:
	uv run python -m encaixe.db.migrate

api:
	uv run uvicorn encaixe.api.app:create_app --factory --port 8100

worker:
	uv run arq encaixe.workers.settings.WorkerSettings

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

eval:
	uv run python evals/retrieval.py --app $(APP)

bakeoff:
	uv run python scripts/bakeoff.py

hooks:
	git config core.hooksPath scripts/git-hooks
