.PHONY: up down models migrate api worker test test-integration lint eval bakeoff hooks

up:
	docker compose up -d --wait

down:
	docker compose down

models:
	scripts/pull-models.sh

migrate:
	uv run encaixe migrate

# Em execução nada é baixado nem reportado (modelos HF vêm do `make models`).
OFFLINE = HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1

# Só no loopback por padrão. Backends em outras máquinas: `make api API_HOST=0.0.0.0` (README).
API_HOST ?= 127.0.0.1

api:
	$(OFFLINE) uv run uvicorn encaixe.api.app:create_app --factory --host $(API_HOST) --port 8100

worker:
	$(OFFLINE) uv run arq encaixe.workers.settings.WorkerSettings

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

APP ?= exemplo
PROMPTFOO = scripts/promptfoo.sh

# Precisa da stack de pé (API :8100, worker, adaptador MCP) e de ENCAIXE_EVAL_API_KEY
# (veja `uv run encaixe seed --app $(APP)`). O promptfoo sai com código != 0 quando há casos
# reprovados; quem julga a regressão é o compare_baseline.py (último passo).
# Limpa antes o cache de respostas (enc:resp:*) de TODAS as apps nesse Redis: senão o eval
# mede respostas guardadas (TTL de 1 h), que não passam pelo pipeline atual.
eval:
	uv run encaixe cache clear
	rm -f evals/results/$(APP).json evals/results/$(APP)-retrieval.json
	uv run python evals/multiturn.py --app $(APP) --seed-only
	-$(PROMPTFOO) eval -c apps/$(APP)/evals/promptfooconfig.yaml -o evals/results/$(APP).json
	uv run python evals/multiturn.py --app $(APP)
	$(OFFLINE) uv run python evals/retrieval.py --app $(APP)
	uv run python evals/compare_baseline.py --app $(APP)

bakeoff:
	uv run python evals/multiturn.py --app $(APP) --seed-only
	uv run python scripts/bakeoff.py --app $(APP) $(if $(MODELS),--models $(MODELS))

hooks:
	git config core.hooksPath scripts/git-hooks
