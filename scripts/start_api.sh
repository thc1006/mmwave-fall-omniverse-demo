#!/bin/bash
# =============================================================================
# start_api.sh - Start the FastAPI prediction server
# =============================================================================
#
# Usage:
#   ./scripts/start_api.sh [OPTIONS]
#
# Options:
#   --host HOST    Host to bind (default: 0.0.0.0)
#   --port PORT    Port to bind (default: 8000)
#   --reload       Enable auto-reload for development
#   --help         Show this help message
#
# =============================================================================

set -e

# Default values
HOST="0.0.0.0"
PORT=8000
RELOAD=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --reload)
            RELOAD="--reload"
            shift
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
echo "Starting mmWave Fall Detection API"
echo "============================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "============================================="
echo ""
echo "API Documentation: http://$HOST:$PORT/docs"
echo "Health Check:      http://$HOST:$PORT/health"
echo "WebSocket:         ws://$HOST:$PORT/ws/events"
echo ""

uvicorn services.api.main:app --host "$HOST" --port "$PORT" $RELOAD
