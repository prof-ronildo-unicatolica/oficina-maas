# Atalhos da oficina. Sem make? Copie o comando da direita.
JOBS = docker compose run --rm jobs python -m
RUFF = docker run --rm --user "$$(id -u):$$(id -g)" -e RUFF_NO_CACHE=true -v "$(CURDIR)":/io -w /io ghcr.io/astral-sh/ruff:0.15.11

# --- fluxo da oficina ---
build:   ; docker compose build
train:   ; $(JOBS) src.train
up:      ; docker compose up -d api
normal:  ; $(JOBS) src.simulate --lote normal
crise:   ; $(JOBS) src.simulate --lote crise
monitor: ; $(JOBS) src.monitor
retrain: ; $(JOBS) src.retrain
logs:    ; docker compose logs -f api
down:    ; docker compose down

# Apaga tudo o que a oficina gerou (lotes, modelos, logs, relatórios) e para os containers.
# Mantém data/credit_g.csv e data/LEIA-ME.txt, que vêm com o projeto.
reset:   ; docker compose run --rm --entrypoint sh jobs -c "cd /app/data && find . -type f ! -name credit_g.csv ! -name LEIA-ME.txt -delete; rm -rf /app/models/* /app/logs/* /app/reports/*"; docker compose down

# --- qualidade de código (Ruff, sem instalar nada: roda num container) ---
lint:    ; $(RUFF) check src && $(RUFF) format --check src
format:  ; $(RUFF) check --fix src && $(RUFF) format src

.PHONY: build train up normal crise monitor retrain logs down reset lint format
