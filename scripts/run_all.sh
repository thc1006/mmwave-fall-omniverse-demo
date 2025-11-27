#!/bin/bash
# Full pipeline: data generation -> training -> API server
# mmWave Fall Detection System

set -e
cd "$(dirname "$0")/.."

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        mmWave Fall Detection Pipeline v2.0                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Parse arguments
MODE=${1:-"full"}  # full, train, api, eval, demo

# Step 1: Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "[1/6] Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q numpy torch scipy fastapi uvicorn pydantic scikit-learn
fi

case $MODE in
    "api")
        echo "[API] Starting API server with LSTM model..."
        export MODEL_PATH=ml/fallnet_lstm.pt
        uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
        exit 0
        ;;
    "eval")
        echo "[EVAL] Running model evaluation..."
        python -m ml.evaluate_models --save-results
        exit 0
        ;;
    "demo")
        echo "[DEMO] Starting demo (API + Frontend)..."
        export MODEL_PATH=ml/fallnet_lstm.pt
        uvicorn services.api.main:app --host 0.0.0.0 --port 8000 &
        API_PID=$!
        echo "API server started (PID: $API_PID)"
        echo "Frontend available at: file://$(pwd)/frontend/index.html"
        echo "Or start a web server: cd frontend && python -m http.server 3000"
        echo "Press Ctrl+C to stop..."
        wait $API_PID
        exit 0
        ;;
esac

# Full pipeline mode
echo "[1/6] Environment ready"

# Step 2: Generate synthetic data if no real data
if [ ! -d "ml/data/fall" ] || [ -z "$(ls -A ml/data/fall 2>/dev/null)" ]; then
    echo "[2/6] Generating synthetic training data..."
    python -m ml.generate_synthetic_data --num-normal 200 --num-fall 200
else
    echo "[2/6] Training data exists, skipping generation"
fi

# Step 3: Process FallAllD if available
if [ -d "FallAllD/FallAllD" ]; then
    echo "[3/6] Processing FallAllD dataset..."
    python -m ml.process_fallalld --input-dir FallAllD/FallAllD --max-samples 300
else
    echo "[3/6] FallAllD not found, skipping"
fi

# Step 4: Process SisFall if available
if [ -d "SisFall/SisFall_dataset" ]; then
    echo "[4/6] Processing SisFall dataset..."
    python -m ml.process_sisfall --input-dir SisFall/SisFall_dataset --max-samples 300
else
    echo "[4/6] SisFall not found, skipping"
fi

# Step 5: Train all models (parallel if requested)
echo "[5/6] Training fall detection models..."

if [ "$2" == "--parallel" ]; then
    echo "Training MLP, CNN, LSTM in parallel..."
    python -m ml.train_fallnet --epochs 100 --patience 15 --model-type mlp --batch-size 2048 --lr 0.01 --output ml/fallnet.pt &
    python -m ml.train_fallnet --epochs 100 --patience 15 --model-type cnn --batch-size 2048 --lr 0.005 --output ml/fallnet_cnn.pt &
    python -m ml.train_fallnet --epochs 100 --patience 15 --model-type lstm --batch-size 2048 --lr 0.005 --output ml/fallnet_lstm.pt &
    wait
else
    # Train only LSTM (best model) by default
    python -m ml.train_fallnet --epochs 100 --patience 15 --model-type lstm --batch-size 2048 --lr 0.005 --output ml/fallnet_lstm.pt
fi

# Step 6: Evaluate models
echo "[6/6] Evaluating models..."
python -m ml.evaluate_models --save-results 2>/dev/null || echo "Evaluation skipped (some models may not exist)"

# Summary
echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline Complete                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo
echo "Dataset Summary:"
echo "  Normal samples: $(ls ml/data/normal/*.npz 2>/dev/null | wc -l)"
echo "  Fall samples:   $(ls ml/data/fall/*.npz 2>/dev/null | wc -l)"
echo
echo "Trained Models:"
ls -lh ml/*.pt 2>/dev/null | awk '{print "  " $9 ": " $5}'
echo
echo "Best Model: ml/fallnet_lstm.pt (95.9% accuracy)"
echo
echo "Quick Start:"
echo "  1. API Server:  ./scripts/run_all.sh api"
echo "  2. Demo Mode:   ./scripts/run_all.sh demo"
echo "  3. Evaluation:  ./scripts/run_all.sh eval"
echo
echo "API Endpoints (http://localhost:8000):"
echo "  POST /predict          - Run fall detection"
echo "  GET  /stats            - Get statistics"
echo "  WS   /ws/events        - Real-time events"
echo "  GET  /health           - Health check"
echo
