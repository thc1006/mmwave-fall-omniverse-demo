# Isaac Sim Streaming Troubleshooting Guide

This document provides solutions for common Isaac Sim WebRTC and Native Livestream issues.

## Error: 0x800E8401 - Net Stream Creation failed

This is the most common WebRTC streaming error. It typically occurs due to:

### 1. Missing GPU Capabilities

**Solution:** Ensure proper NVIDIA container runtime configuration.

```bash
# Check NVIDIA container runtime
docker info | grep -i nvidia

# If not present, install nvidia-container-toolkit:
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Port Conflicts

WebRTC requires multiple ports. Check if they're available:

```bash
# Check port availability
netstat -tuln | grep -E '8211|8011|8012|4799[5-9]|4800[0-9]|4801[0-2]'

# Kill processes using required ports if needed
sudo fuser -k 8211/tcp
```

### 3. Network Mode Issues

WebRTC works best with `network_mode: host`. If you can't use host mode:

```yaml
# Alternative port configuration (in docker-compose)
ports:
  - "8211:8211/tcp"
  - "8211:8211/udp"  # WebRTC needs UDP!
  - "8011:8011"
  - "8012:8012"
  - "47995-48012:47995-48012/tcp"
  - "47995-48012:47995-48012/udp"
```

### 4. Vulkan/GPU Issues

```bash
# Check Vulkan availability inside container
docker exec isaac-streaming vulkaninfo --summary

# If Vulkan fails, ensure proper ICD file path:
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### 5. Memory Limits

Isaac Sim requires significant GPU memory. Check availability:

```bash
nvidia-smi --query-gpu=memory.free --format=csv,noheader

# If memory is low, stop other GPU processes
```

## Error: Stream server initialization failed

### Solution: Use Native Livestream Instead

If WebRTC continues to fail, use Native Livestream:

```bash
# Start Native Livestream
make stream-native

# Or directly:
docker compose -f infra/docker-compose.isaac-native-streaming.yml up -d
```

## Common Configuration Issues

### 1. EULA Not Accepted

```yaml
environment:
  - ACCEPT_EULA=Y
  - PRIVACY_CONSENT=Y  # Also required!
```

### 2. Root Execution

Isaac Sim 4.2.0 may need root permissions:

```yaml
environment:
  - OMNI_KIT_ALLOW_ROOT=1
command:
  - ... --allow-root
```

### 3. Display/Graphics Issues

For headless operation:

```yaml
environment:
  - NVIDIA_DRIVER_CAPABILITIES=all,graphics,display
```

## Checking Streaming Status

### WebRTC

```bash
# Check if WebRTC demo page is accessible
curl -f http://localhost:8211/streaming/webrtc-demo

# Check container health
docker inspect --format='{{.State.Health.Status}}' chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1
```

### Native Livestream

```bash
# Check Kit HTTP server
curl -f http://localhost:8011/status

# Check streaming status
curl http://localhost:8011/streaming/status
```

## Log Analysis

```bash
# View real-time logs
docker logs -f chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1

# Search for specific errors
docker logs chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1 2>&1 | grep -i "error\|fail\|warn"

# Check Omniverse logs inside container
docker exec chih-tu-qi-ltc-mmwave-sim-isaac-streaming-1 \
  cat /root/.nvidia-omniverse/logs/Kit/Isaac-Sim/*/kit_*.log | tail -100
```

## Alternative Streaming Methods

### 1. VNC-based Approach

If WebRTC/Native streaming continues to fail, use VNC:

```yaml
# Add to docker-compose
services:
  isaac-vnc:
    image: nvcr.io/nvidia/isaac-sim:4.2.0
    environment:
      - DISPLAY=:1
      - VNC_PASSWORD=isaac123
    ports:
      - "5901:5901"
    command: |
      Xvfb :1 -screen 0 1920x1080x24 &
      x11vnc -display :1 -forever -rfbport 5901 -passwd isaac123 &
      /isaac-sim/isaac-sim.sh
```

### 2. Screenshot-based Monitoring

For simple monitoring without real-time streaming:

```python
# Inside Isaac Sim
import omni.kit.viewport.utility as viewport_utils
viewport = viewport_utils.get_active_viewport()
viewport.capture_frame("screenshot.png")
```

## Quick Fixes Checklist

1. [ ] NVIDIA container runtime installed
2. [ ] `ACCEPT_EULA=Y` and `PRIVACY_CONSENT=Y` set
3. [ ] `network_mode: host` enabled
4. [ ] `ipc: host` enabled
5. [ ] GPU has sufficient free memory (>8GB)
6. [ ] Ports 8211, 8011, 8012 are available
7. [ ] `--allow-root` flag used
8. [ ] Vulkan ICD path is correct

## Getting Help

If issues persist:

1. Check NVIDIA Isaac Sim documentation: https://docs.omniverse.nvidia.com/isaacsim/
2. NVIDIA Developer Forums: https://forums.developer.nvidia.com/c/omniverse/
3. Check container logs for specific error messages

## Version-Specific Notes

### Isaac Sim 4.2.0

- WebRTC extension path: `omni.services.streamclient.webrtc`
- Native Livestream extension: `omni.kit.livestream.native`
- Use `runheadless.webrtc.sh` or `runheadless.native.sh` scripts
- May require `--/app/livestream/logLevel=verbose` for debugging
