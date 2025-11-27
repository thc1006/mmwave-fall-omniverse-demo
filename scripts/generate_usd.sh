#!/bin/bash
# =============================================================================
# generate_usd.sh - Generate USD scene from YAML configuration
# =============================================================================
#
# Usage:
#   ./scripts/generate_usd.sh [OPTIONS]
#
# Options:
#   --config FILE    YAML configuration file (default: facility/chih_tu_qi_floor1_ltc.yaml)
#   --output FILE    Output USD file (default: sim/usd/chih_tu_qi_floor1_ltc.usd)
#   --help           Show this help message
#
# =============================================================================

set -e

# Default values
CONFIG="facility/chih_tu_qi_floor1_ltc.yaml"
OUTPUT="sim/usd/chih_tu_qi_floor1_ltc.usd"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --help)
            head -15 "$0" | tail -n +2 | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================="
echo "Generating USD Scene from YAML"
echo "============================================="
echo "Config: $CONFIG"
echo "Output: $OUTPUT"
echo "============================================="

# Create output directory
mkdir -p "$(dirname "$OUTPUT")"

# Run generator
python3 sim/usd/generate_floor1_from_yaml.py \
    --config "$CONFIG" \
    --out "$OUTPUT"

echo ""
echo "USD scene generated: $OUTPUT"
