# =============================================================================
# 赤土崎多功能館 mmWave Fall Detection Demo - Makefile
# =============================================================================

.PHONY: help install dev api api-dev docker-up docker-down generate-usd sim-job train test clean \
       frontend

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║     赤土崎多功能館 mmWave Fall Detection Makefile            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install        - Install Python dependencies"
	@echo "    dev            - Install development dependencies"
	@echo ""
	@echo "  API Server:"
	@echo "    api            - Start API server (production)"
	@echo "    api-dev        - Start API server with hot reload"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-up      - Start Isaac Sim + API Docker containers"
	@echo "    docker-down    - Stop Docker containers"
	@echo "    docker-logs    - View Docker logs"
	@echo ""
	@echo "  Simulation:"
	@echo "    generate-usd   - Generate USD scene from YAML config"
	@echo "    sim-job        - Run simulation job (SCENARIO=xxx EPISODES=n)"
	@echo "    sim-fall       - Run fall detection simulation"
	@echo "    sim-normal     - Run normal activity simulation"
	@echo ""
	@echo "  ML Training:"
	@echo "    train          - Train FallNet model (full)"
	@echo "    train-quick    - Quick training (10 epochs)"
	@echo ""
	@echo "  Frontend:"
	@echo "    frontend       - Start frontend dev server"
	@echo ""
	@echo "  Testing:"
	@echo "    test           - Run all tests"
	@echo "    test-api       - Test API endpoints"
	@echo ""
	@echo "  Utilities:"
	@echo "    clean          - Clean generated files"
	@echo "    lint           - Run linter"
	@echo "    format         - Format code"
	@echo ""
	@echo "Examples:"
	@echo "  make api-dev"
	@echo "  make generate-usd"
	@echo "  make sim-job SCENARIO=fall EPISODES=100"
	@echo "  make train"

# =============================================================================
# Environment Variables
# =============================================================================

PYTHON ?= python3
VENV ?= .venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python

# API Settings
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
MODEL_PATH ?= ml/fallnet_lstm.pt

# Simulation Settings
SCENARIO ?= normal
EPISODES ?= 100
DURATION ?= 5.0
OUTPUT_DIR ?= ml/data
CONFIG_FILE ?= facility/chih_tu_qi_floor1_ltc.yaml
USD_OUTPUT ?= sim/usd/chih_tu_qi_floor1_ltc.usd

# =============================================================================
# Installation
# =============================================================================

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

dev: install
	$(PIP) install pytest pytest-cov black ruff mypy httpx

# =============================================================================
# API Server
# =============================================================================

api: install
	@echo "Starting API server on $(API_HOST):$(API_PORT)..."
	MODEL_PATH=$(MODEL_PATH) $(VENV)/bin/uvicorn \
		services.api.main:app \
		--host $(API_HOST) \
		--port $(API_PORT)

api-dev: install
	@echo "Starting API server with hot reload on $(API_HOST):$(API_PORT)..."
	@echo "API Documentation: http://localhost:$(API_PORT)/docs"
	MODEL_PATH=$(MODEL_PATH) $(VENV)/bin/uvicorn \
		services.api.main:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--reload

api-bg: install
	@echo "Starting API server in background..."
	MODEL_PATH=$(MODEL_PATH) nohup $(VENV)/bin/uvicorn \
		services.api.main:app \
		--host $(API_HOST) \
		--port $(API_PORT) > /tmp/api.log 2>&1 &
	@echo "API server started. Logs: /tmp/api.log"

# =============================================================================
# Docker
# =============================================================================

docker-up:
	@echo "Starting Docker containers..."
	docker compose -f infra/docker-compose.isaac-headless.yml up -d
	@echo "Containers started. Use 'make docker-logs' to view logs."

docker-down:
	@echo "Stopping Docker containers..."
	docker compose -f infra/docker-compose.isaac-headless.yml down

docker-logs:
	docker compose -f infra/docker-compose.isaac-headless.yml logs -f

docker-status:
	docker compose -f infra/docker-compose.isaac-headless.yml ps

# =============================================================================
# USD Scene Generation
# =============================================================================

generate-usd: install
	@echo "Generating USD scene from $(CONFIG_FILE)..."
	@mkdir -p $(dir $(USD_OUTPUT))
	$(PYTHON_VENV) sim/usd/generate_floor1_from_yaml.py \
		--config $(CONFIG_FILE) \
		--out $(USD_OUTPUT)
	@echo "USD scene generated: $(USD_OUTPUT)"

# =============================================================================
# Simulation Jobs
# =============================================================================

sim-job:
	@echo "Running simulation job: scenario=$(SCENARIO), episodes=$(EPISODES)"
	./scripts/run_sim_job.sh \
		--scenario $(SCENARIO) \
		--episodes $(EPISODES) \
		--duration $(DURATION) \
		--output-dir $(OUTPUT_DIR)

sim-fall:
	$(MAKE) sim-job SCENARIO=fall EPISODES=$(EPISODES)

sim-normal:
	$(MAKE) sim-job SCENARIO=normal EPISODES=$(EPISODES)

sim-rehab:
	$(MAKE) sim-job SCENARIO=rehab_bad_posture EPISODES=$(EPISODES)

sim-chest:
	$(MAKE) sim-job SCENARIO=chest_abnormal EPISODES=$(EPISODES)

sim-all: sim-normal sim-fall sim-rehab sim-chest
	@echo "All scenarios simulated!"

# =============================================================================
# ML Training
# =============================================================================

train: install
	@echo "Training FallNet model..."
	$(PYTHON_VENV) ml/train_fallnet.py \
		--data-dir $(OUTPUT_DIR) \
		--output $(MODEL_PATH) \
		--epochs 100 \
		--batch-size 32

train-quick: install
	@echo "Quick training (10 epochs)..."
	$(PYTHON_VENV) ml/train_fallnet.py \
		--data-dir $(OUTPUT_DIR) \
		--output $(MODEL_PATH) \
		--epochs 10

train-lstm: install
	$(PYTHON_VENV) ml/train_fallnet.py \
		--data-dir $(OUTPUT_DIR) \
		--output ml/fallnet_lstm.pt \
		--model-type lstm \
		--epochs 100

train-cnn: install
	$(PYTHON_VENV) ml/train_fallnet.py \
		--data-dir $(OUTPUT_DIR) \
		--output ml/fallnet_cnn.pt \
		--model-type cnn \
		--epochs 100

# =============================================================================
# Frontend
# =============================================================================

frontend:
	@echo "Starting frontend dev server..."
	cd frontend && npm install && npm run dev

frontend-build:
	cd frontend && npm install && npm run build

# =============================================================================
# Testing
# =============================================================================

test: install
	$(PYTHON_VENV) -m pytest tests/ -v

test-api: install
	@echo "Testing API health..."
	@curl -s http://localhost:$(API_PORT)/health | python -m json.tool || echo "API not running"
	@echo ""
	@echo "Testing /stats endpoint..."
	@curl -s http://localhost:$(API_PORT)/stats | python -m json.tool || true

test-predict: install
	@echo "Testing /predict endpoint..."
	@curl -X POST http://localhost:$(API_PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"sequences": [{"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]}]}' \
		| python -m json.tool || echo "API not running or prediction failed"

# =============================================================================
# Code Quality
# =============================================================================

lint: dev
	$(VENV)/bin/ruff check .

format: dev
	$(VENV)/bin/black .

typecheck: dev
	$(VENV)/bin/mypy services/ ml/ --ignore-missing-imports

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info
	find . -name "*.pyc" -delete

clean-usd:
	rm -f sim/usd/*.usd

clean-data:
	rm -rf ml/data/*/*.npz ml/data/*/*.npy

clean-all: clean clean-usd
	rm -rf $(VENV)
	rm -f ml/*.pt

# =============================================================================
# Full Pipeline
# =============================================================================

pipeline: install generate-usd sim-all train
	@echo "Full pipeline complete!"
	@echo "Run 'make api' to start the prediction server."

all: install generate-usd train api-bg
	@echo "Setup complete!"
	@echo "API running at http://localhost:$(API_PORT)"
	@echo "Documentation at http://localhost:$(API_PORT)/docs"
