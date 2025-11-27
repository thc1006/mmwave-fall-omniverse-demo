#!/bin/bash
# =============================================================================
# run_sim_job.sh - Run Isaac Sim simulation job for mmWave fall detection
# =============================================================================
#
# Usage:
#   ./scripts/run_sim_job.sh [OPTIONS]
#
# Options:
#   --scenario SCENARIO   Scenario to simulate (normal|fall|rehab_bad_posture|chest_abnormal)
#   --episodes N          Number of episodes to record (default: 100)
#   --duration D          Duration per episode in seconds (default: 5.0)
#   --output-dir DIR      Output directory for recorded data (default: ml/data)
#   --headless            Run in headless mode
#   --docker              Run inside Docker container
#   --help                Show this help message
#
# Examples:
#   ./scripts/run_sim_job.sh --scenario fall --episodes 50
#   ./scripts/run_sim_job.sh --scenario normal --episodes 100 --headless
#   ./scripts/run_sim_job.sh --docker --scenario fall --episodes 200
#
# =============================================================================

set -e

# Default values
SCENARIO="normal"
EPISODES=100
DURATION=5.0
OUTPUT_DIR="ml/data"
HEADLESS=false
USE_DOCKER=false
ISAAC_SIM_PATH="${ISAAC_SIM_PATH:-/home/thc1006/.local/share/ov/pkg/isaac_sim-2023.1.1}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --docker)
            USE_DOCKER=true
            shift
            ;;
        --help)
            head -30 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate scenario
case $SCENARIO in
    normal|fall|rehab_bad_posture|chest_abnormal)
        ;;
    *)
        echo "Error: Invalid scenario '$SCENARIO'"
        echo "Valid scenarios: normal, fall, rehab_bad_posture, chest_abnormal"
        exit 1
        ;;
esac

echo "============================================="
echo "mmWave Fall Detection - Simulation Job"
echo "============================================="
echo "Scenario:    $SCENARIO"
echo "Episodes:    $EPISODES"
echo "Duration:    ${DURATION}s per episode"
echo "Output:      $OUTPUT_DIR"
echo "Headless:    $HEADLESS"
echo "Docker:      $USE_DOCKER"
echo "============================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build Python command
PYTHON_CMD="
import sys
sys.path.insert(0, '.')

from sim.mmwave_fall_extension.record_fall_data import RecordFallDataScript

script = RecordFallDataScript()
script.run(
    num_episodes=$EPISODES,
    scenario='$SCENARIO',
    duration_per_episode=$DURATION,
    output_dir='$OUTPUT_DIR',
)
"

if [ "$USE_DOCKER" = true ]; then
    echo "Running inside Docker container..."
    docker compose -f infra/docker-compose.isaac-headless.yml run --rm isaac-sim \
        ./python.sh -c "$PYTHON_CMD"
else
    if [ -d "$ISAAC_SIM_PATH" ]; then
        echo "Running with Isaac Sim at: $ISAAC_SIM_PATH"

        if [ "$HEADLESS" = true ]; then
            HEADLESS_ARG="--headless"
        else
            HEADLESS_ARG=""
        fi

        cd "$ISAAC_SIM_PATH"
        ./python.sh $HEADLESS_ARG -c "$PYTHON_CMD"
    else
        echo "Warning: Isaac Sim not found at $ISAAC_SIM_PATH"
        echo "Running in standalone mode with synthetic data..."
        python3 -c "$PYTHON_CMD"
    fi
fi

echo ""
echo "============================================="
echo "Simulation job completed!"
echo "Data saved to: $OUTPUT_DIR"
echo "============================================="
