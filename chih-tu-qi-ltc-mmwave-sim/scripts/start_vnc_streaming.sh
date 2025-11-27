#!/bin/bash
# Isaac Sim VNC Streaming Script
# Provides browser-based access to Isaac Sim 3D rendering via noVNC

set -e

DISPLAY_NUM=99
SCREEN_RES="1920x1080x24"
VNC_PORT=5900
NOVNC_PORT=6080
ISAAC_SIM_DIR="/home/thc1006/isaac-sim"

echo "=============================================="
echo " Isaac Sim VNC Streaming Setup"
echo "=============================================="

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "x11vnc.*:${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
sleep 2

# Start Xvfb (virtual display)
echo "Starting virtual display :${DISPLAY_NUM}..."
Xvfb :${DISPLAY_NUM} -screen 0 ${SCREEN_RES} &
XVFB_PID=$!
sleep 2

# Verify Xvfb is running
if ! ps -p $XVFB_PID > /dev/null; then
    echo "ERROR: Xvfb failed to start"
    exit 1
fi
echo "Virtual display started (PID: $XVFB_PID)"

# Start x11vnc
echo "Starting VNC server on port ${VNC_PORT}..."
x11vnc -display :${DISPLAY_NUM} -forever -nopw -shared -rfbport ${VNC_PORT} -bg -o /tmp/x11vnc.log 2>&1
sleep 2

# Start noVNC (browser-based VNC client)
echo "Starting noVNC on port ${NOVNC_PORT}..."
websockify --web=/usr/share/novnc ${NOVNC_PORT} localhost:${VNC_PORT} &
NOVNC_PID=$!
sleep 2

echo ""
echo "=============================================="
echo " VNC Streaming Ready!"
echo "=============================================="
echo ""
echo "Browser access: http://localhost:${NOVNC_PORT}/vnc.html"
echo "VNC port: ${VNC_PORT}"
echo "Virtual display: :${DISPLAY_NUM}"
echo ""
echo "Starting Isaac Sim..."
echo ""

# Export display for Isaac Sim
export DISPLAY=:${DISPLAY_NUM}

# Start Isaac Sim with GUI
cd ${ISAAC_SIM_DIR}
./isaac-sim.sh \
    --/app/window/width=1920 \
    --/app/window/height=1080 \
    --/renderer/activeGpu=0

# Cleanup on exit
cleanup() {
    echo "Shutting down..."
    pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
    pkill -f "x11vnc.*:${DISPLAY_NUM}" 2>/dev/null || true
    pkill -f "websockify.*${NOVNC_PORT}" 2>/dev/null || true
}
trap cleanup EXIT
