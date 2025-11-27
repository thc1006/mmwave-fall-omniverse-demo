#!/bin/bash
# =============================================================================
# train_model.sh - Train FallNet model
# =============================================================================
#
# Usage:
#   ./scripts/train_model.sh [OPTIONS]
#
# Options:
#   --data-dir DIR      Input data directory (default: ml/data)
#   --output FILE       Output model file (default: ml/fallnet_lstm.pt)
#   --model-type TYPE   Model type: mlp|cnn|lstm (default: lstm)
#   --epochs N          Number of training epochs (default: 100)
#   --batch-size N      Batch size (default: 32)
#   --learning-rate LR  Learning rate (default: 0.001)
#   --help              Show this help message
#
# =============================================================================

set -e

# Default values
DATA_DIR="ml/data"
OUTPUT="ml/fallnet_lstm.pt"
MODEL_TYPE="lstm"
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --help)
            head -18 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "Training FallNet Model"
echo "============================================="
echo "Data:          $DATA_DIR"
echo "Output:        $OUTPUT"
echo "Model Type:    $MODEL_TYPE"
echo "Epochs:        $EPOCHS"
echo "Batch Size:    $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "============================================="

# Run training
python3 ml/train_fallnet.py \
    --data-dir "$DATA_DIR" \
    --output "$OUTPUT" \
    --model-type "$MODEL_TYPE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE"

echo ""
echo "Model saved to: $OUTPUT"
