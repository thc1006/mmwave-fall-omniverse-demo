# =============================================================================
# 赤土崎多功能館 mmWave Fall Detection - Makefile
# =============================================================================

.PHONY: help install dev api api-dev docker-up docker-down generate-usd sim-job train test clean \
       stream-webrtc stream-native stream-stop stream-logs

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║     赤土崎多功能館 mmWave Fall Detection Makefile            ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  install        - Install Python dependencies"
	@echo "  dev            - Install development dependencies"
	@echo "  api            - Start API server (production)"
	@echo "  api-dev        - Start API server with hot reload"
	@echo "  docker-up      - Start Isaac Sim Docker container"
	@echo "  docker-down    - Stop Isaac Sim Docker container"
	@echo "  stream-webrtc  - Start WebRTC streaming (browser)"
	@echo "  stream-native  - Start Native Livestream"
	@echo "  stream-stop    - Stop streaming containers"
	@echo "  stream-logs    - View streaming logs"
	@echo "  generate-usd   - Generate USD scene from YAML config"
	@echo "  sim-job        - Run simulation job (use SCENARIO=xxx)"
	@echo "  train          - Train FallNet model"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make api-dev"
	@echo "  make docker-up && make sim-job SCENARIO=fall_incident"
	@echo "  make generate-usd"

# =============================================================================
# Environment
# =============================================================================

PYTHON ?= python3
VENV ?= .venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python

# API Settings
API_HOST ?= 0.0.0.0
API_PORT ?= 8000
MODEL_PATH ?= ml/fallnet.pt

# Simulation Settings
SCENARIO ?= normal_operation
EPISODES ?= 10
FRAMES ?= 128
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
	$(PIP) install -r requirements.txt 2>/dev/null || $(PIP) install \
		torch numpy scipy fastapi uvicorn pydantic pyyaml

dev: install
	$(PIP) install pytest pytest-cov black ruff mypy

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
# Docker / Isaac Sim
# =============================================================================

docker-up:
	@echo "Starting Isaac Sim Docker container..."
	docker compose -f infra/docker-compose.isaac-headless.yml up -d
	@echo "Container started. Use 'make docker-logs' to view logs."

docker-down:
	@echo "Stopping Isaac Sim Docker container..."
	docker compose -f infra/docker-compose.isaac-headless.yml down

docker-logs:
	docker compose -f infra/docker-compose.isaac-headless.yml logs -f

docker-status:
	docker compose -f infra/docker-compose.isaac-headless.yml ps

# =============================================================================
# Streaming (WebRTC / Native Livestream)
# =============================================================================

STREAM_PORT ?= 8211
STREAM_WIDTH ?= 1920
STREAM_HEIGHT ?= 1080

stream-webrtc:
	@echo "Starting Isaac Sim WebRTC streaming..."
	./infra/scripts/start_streaming.sh webrtc --port $(STREAM_PORT) \
		--width $(STREAM_WIDTH) --height $(STREAM_HEIGHT)

stream-webrtc-scene:
	@echo "Starting Isaac Sim WebRTC streaming with scene..."
	./infra/scripts/start_streaming.sh webrtc --load-scene \
		--scene $(USD_OUTPUT) --port $(STREAM_PORT)

stream-native:
	@echo "Starting Isaac Sim Native Livestream..."
	./infra/scripts/start_streaming.sh native --port $(STREAM_PORT) \
		--width $(STREAM_WIDTH) --height $(STREAM_HEIGHT)

stream-native-scene:
	@echo "Starting Isaac Sim Native Livestream with scene..."
	./infra/scripts/start_streaming.sh native --load-scene \
		--scene $(USD_OUTPUT) --port $(STREAM_PORT)

stream-stop:
	@echo "Stopping streaming containers..."
	docker compose -f infra/docker-compose.isaac-streaming.yml down 2>/dev/null || true
	docker compose -f infra/docker-compose.isaac-native-streaming.yml down 2>/dev/null || true

stream-logs:
	@if docker ps --format '{{.Names}}' | grep -q "isaac-streaming"; then \
		docker compose -f infra/docker-compose.isaac-streaming.yml logs -f; \
	elif docker ps --format '{{.Names}}' | grep -q "isaac-native-streaming"; then \
		docker compose -f infra/docker-compose.isaac-native-streaming.yml logs -f; \
	else \
		echo "No streaming container running."; \
	fi

stream-load-scene:
	@echo "Loading scene into running streaming container..."
	@if docker ps --format '{{.Names}}' | grep -q "isaac-streaming"; then \
		docker exec -e USD_STAGE_PATH="/workspace/chih-tu-qi-ltc-mmwave-sim/$(USD_OUTPUT)" \
			chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1 \
			/isaac-sim/python.sh /workspace/chih-tu-qi-ltc-mmwave-sim/sim/scripts/load_scene_streaming.py; \
	elif docker ps --format '{{.Names}}' | grep -q "isaac-native-streaming"; then \
		docker exec -e USD_STAGE_PATH="/workspace/chih-tu-qi-ltc-mmwave-sim/$(USD_OUTPUT)" \
			chih-tu-qi-ltc-mmwave-sim-isaac-native-streaming-1 \
			/isaac-sim/python.sh /workspace/chih-tu-qi-ltc-mmwave-sim/sim/scripts/load_scene_streaming.py; \
	else \
		echo "No streaming container running. Start with 'make stream-webrtc' first."; \
	fi

# =============================================================================
# USD Generation
# =============================================================================

generate-usd: install
	@echo "Generating USD scene from $(CONFIG_FILE)..."
	$(PYTHON_VENV) sim/usd/generate_floor1_from_yaml.py \
		--config $(CONFIG_FILE) \
		--out $(USD_OUTPUT)

generate-usd-docker:
	@echo "Generating USD scene inside Docker container..."
	docker exec chih-tu-qi-ltc-mmwave-sim-isaac-headless-1 \
		/isaac-sim/python.sh sim/usd/generate_floor1_from_yaml.py \
		--config $(CONFIG_FILE) \
		--out $(USD_OUTPUT)

# =============================================================================
# Simulation
# =============================================================================

sim-job:
	@echo "Running simulation job: $(SCENARIO)"
	./infra/scripts/run_sim_job.sh $(SCENARIO) \
		--episodes $(EPISODES) \
		--frames $(FRAMES) \
		--output $(OUTPUT_DIR)

sim-normal:
	$(MAKE) sim-job SCENARIO=normal_operation

sim-fall:
	$(MAKE) sim-job SCENARIO=fall_incident

sim-wandering:
	$(MAKE) sim-job SCENARIO=wandering_alert

# =============================================================================
# Training
# =============================================================================

train: install
	@echo "Training FallNet model..."
	$(PYTHON_VENV) -m ml.train_fallnet \
		--data-dir ml/data \
		--output $(MODEL_PATH) \
		--epochs 100 \
		--patience 15

train-quick: install
	@echo "Quick training (10 epochs)..."
	$(PYTHON_VENV) -m ml.train_fallnet \
		--data-dir ml/data \
		--output $(MODEL_PATH) \
		--epochs 10 \
		--patience 5

# =============================================================================
# Testing
# =============================================================================

test: install
	$(PYTHON_VENV) -m pytest tests/ -v

test-api: install
	@echo "Testing API health..."
	curl -s http://localhost:$(API_PORT)/health | python -m json.tool
	@echo ""
	@echo "Testing /info endpoint..."
	curl -s http://localhost:$(API_PORT)/info | python -m json.tool

test-predict: install
	@echo "Testing /predict endpoint..."
	curl -X POST http://localhost:$(API_PORT)/predict \
		-H "Content-Type: application/json" \
		-d '{"sequences": [[0.1, 0.2, 0.3, 0.4, 0.5]]}' | python -m json.tool

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -f sim/usd/*.usd
	rm -f ml/data/*/*.npz

clean-all: clean
	rm -rf $(VENV)
	rm -f ml/*.pt

# =============================================================================
# Development Utilities
# =============================================================================

lint: dev
	$(VENV)/bin/ruff check .

format: dev
	$(VENV)/bin/black .

typecheck: dev
	$(VENV)/bin/mypy services/ ml/

# =============================================================================
# Full Pipeline
# =============================================================================

all: install generate-usd train api-bg
	@echo "Full pipeline complete!"
	@echo "API running at http://localhost:$(API_PORT)"
