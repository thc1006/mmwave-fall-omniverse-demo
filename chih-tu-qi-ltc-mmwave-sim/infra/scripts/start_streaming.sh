#!/usr/bin/env bash
# =============================================================================
# start_streaming.sh - Start Isaac Sim WebRTC/Native Streaming
# =============================================================================
# This script handles starting Isaac Sim with streaming enabled and
# optionally loading a USD scene.
#
# Usage:
#   ./infra/scripts/start_streaming.sh [webrtc|native] [options]
#
# Options:
#   --scene PATH     Path to USD scene file
#   --load-scene     Load scene after streaming starts
#   --port PORT      WebRTC port (default: 8211)
#   --width WIDTH    Viewport width (default: 1920)
#   --height HEIGHT  Viewport height (default: 1080)
#   --help           Show this help message
#
# Examples:
#   ./infra/scripts/start_streaming.sh webrtc
#   ./infra/scripts/start_streaming.sh native --scene sim/usd/chih_tu_qi_floor1_ltc.usd
#   ./infra/scripts/start_streaming.sh webrtc --load-scene
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
blue() { echo -e "${BLUE}$*${NC}"; }

# Default configuration
STREAMING_MODE="${1:-webrtc}"
SCENE_PATH="${SCENE_PATH:-sim/usd/chih_tu_qi_floor1_ltc.usd}"
LOAD_SCENE=false
PORT="${PORT:-8211}"
WIDTH="${WIDTH:-1920}"
HEIGHT="${HEIGHT:-1080}"

# Parse arguments
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --scene)
            SCENE_PATH="$2"
            shift 2
            ;;
        --load-scene)
            LOAD_SCENE=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [webrtc|native] [options]"
            echo ""
            echo "Streaming modes:"
            echo "  webrtc   - Browser-based WebRTC streaming (default)"
            echo "  native   - Omniverse Streaming Client"
            echo ""
            echo "Options:"
            echo "  --scene PATH     Path to USD scene file"
            echo "  --load-scene     Load scene after streaming starts"
            echo "  --port PORT      Streaming port (default: 8211)"
            echo "  --width WIDTH    Viewport width (default: 1920)"
            echo "  --height HEIGHT  Viewport height (default: 1080)"
            echo "  --help           Show this help"
            exit 0
            ;;
        *)
            warn "Unknown argument: $1"
            shift
            ;;
    esac
done

# Validate streaming mode
if [[ "${STREAMING_MODE}" != "webrtc" && "${STREAMING_MODE}" != "native" ]]; then
    error "Invalid streaming mode: ${STREAMING_MODE}. Use 'webrtc' or 'native'."
fi

# Check for NVIDIA GPU
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        error "nvidia-smi not found. NVIDIA GPU required for Isaac Sim streaming."
    fi

    if ! nvidia-smi &> /dev/null; then
        error "Failed to communicate with NVIDIA GPU. Check driver installation."
    fi

    info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
}

# Check Docker setup
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker."
    fi

    if ! docker info &> /dev/null; then
        error "Docker daemon not running or insufficient permissions."
    fi

    # Check nvidia-container-toolkit
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        warn "NVIDIA container runtime may not be configured."
        warn "Install nvidia-container-toolkit if streaming fails."
    fi

    info "Docker is ready"
}

# Start streaming container
start_streaming() {
    local compose_file
    local container_name

    if [[ "${STREAMING_MODE}" == "webrtc" ]]; then
        compose_file="docker-compose.isaac-streaming.yml"
        container_name="chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1"
    else
        compose_file="docker-compose.isaac-native-streaming.yml"
        container_name="chih-tu-qi-ltc-mmwave-sim-isaac-native-streaming-1"
    fi

    info "Starting Isaac Sim ${STREAMING_MODE} streaming..."
    info "Compose file: infra/${compose_file}"

    # Export environment variables for docker-compose
    export WIDTH HEIGHT PORT
    export USD_STAGE_PATH="${PROJECT_ROOT}/${SCENE_PATH}"

    # Stop existing container if running
    docker compose -f "${PROJECT_ROOT}/infra/${compose_file}" down 2>/dev/null || true

    # Start new container
    docker compose -f "${PROJECT_ROOT}/infra/${compose_file}" up -d

    info "Container started. Waiting for Isaac Sim to initialize..."
    info "This may take 2-5 minutes on first run (shader compilation)."

    # Wait for container to be ready
    local max_wait=300
    local waited=0
    local check_interval=10

    while [[ $waited -lt $max_wait ]]; do
        if docker ps --format '{{.Names}}' | grep -q "${container_name}"; then
            # Check if health endpoint is responding
            if [[ "${STREAMING_MODE}" == "webrtc" ]]; then
                if curl -sf "http://localhost:${PORT}/streaming/webrtc-demo" &>/dev/null; then
                    info "WebRTC streaming is ready!"
                    break
                fi
            else
                if curl -sf "http://localhost:8011/status" &>/dev/null; then
                    info "Native streaming is ready!"
                    break
                fi
            fi
        fi

        echo -n "."
        sleep $check_interval
        waited=$((waited + check_interval))
    done
    echo ""

    if [[ $waited -ge $max_wait ]]; then
        warn "Streaming may not be fully ready yet. Check container logs:"
        warn "  docker logs -f ${container_name}"
    fi
}

# Load USD scene
load_scene() {
    local container_name

    if [[ "${STREAMING_MODE}" == "webrtc" ]]; then
        container_name="chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1"
    else
        container_name="chih-tu-qi-ltc-mmwave-sim-isaac-native-streaming-1"
    fi

    info "Loading USD scene: ${SCENE_PATH}"

    # Check if scene file exists
    if [[ ! -f "${PROJECT_ROOT}/${SCENE_PATH}" ]]; then
        warn "Scene file not found: ${SCENE_PATH}"
        warn "You may need to generate it first with:"
        warn "  /isaac-sim/python.sh sim/usd/generate_floor1_from_yaml.py"
        return 1
    fi

    # Execute scene loading script inside container
    docker exec -e USD_STAGE_PATH="/workspace/chih-tu-qi-ltc-mmwave-sim/${SCENE_PATH}" \
        "${container_name}" \
        /isaac-sim/python.sh /workspace/chih-tu-qi-ltc-mmwave-sim/sim/scripts/load_scene_streaming.py &

    info "Scene loading started in background"
}

# Print access information
print_access_info() {
    echo ""
    blue "============================================================"
    blue "  Isaac Sim Streaming Ready"
    blue "============================================================"
    echo ""

    if [[ "${STREAMING_MODE}" == "webrtc" ]]; then
        echo "  WebRTC Streaming URL:"
        echo "    http://localhost:${PORT}/streaming/webrtc-demo"
        echo ""
        echo "  Open this URL in a Chrome/Edge browser to view the stream."
    else
        echo "  Native Livestream:"
        echo "    omniverse://localhost:${PORT}"
        echo ""
        echo "  Use Omniverse Streaming Client to connect."
    fi

    echo ""
    echo "  Container logs:"
    if [[ "${STREAMING_MODE}" == "webrtc" ]]; then
        echo "    docker logs -f chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1"
    else
        echo "    docker logs -f chih-tu-qi-ltc-mmwave-sim-isaac-native-streaming-1"
    fi

    echo ""
    echo "  Stop streaming:"
    if [[ "${STREAMING_MODE}" == "webrtc" ]]; then
        echo "    docker compose -f infra/docker-compose.isaac-streaming.yml down"
    else
        echo "    docker compose -f infra/docker-compose.isaac-native-streaming.yml down"
    fi

    echo ""
    blue "============================================================"
}

# Main execution
main() {
    echo ""
    blue "============================================================"
    blue "  Isaac Sim Streaming Setup"
    blue "  Mode: ${STREAMING_MODE^^}"
    blue "============================================================"
    echo ""

    check_gpu
    check_docker
    start_streaming

    if [[ "${LOAD_SCENE}" == true ]]; then
        load_scene
    fi

    print_access_info
}

main "$@"
