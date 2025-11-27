#!/usr/bin/env python3
"""
Screen capture MJPEG streaming server for Isaac Sim viewport.
Captures the Isaac Sim window and streams it as MJPEG for browser viewing.
"""

import asyncio
import io
import time
from typing import AsyncGenerator
import subprocess
import threading
import queue

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

app = FastAPI(title="Isaac Sim Screen Stream", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global frame buffer
frame_queue: queue.Queue = queue.Queue(maxsize=2)
capture_running = False

def get_isaac_window_geometry() -> tuple[int, int, int, int] | None:
    """Get Isaac Sim window geometry using xdotool/wmctrl."""
    try:
        # Find Isaac Sim window
        result = subprocess.run(
            ["wmctrl", "-l"],
            capture_output=True,
            text=True
        )

        for line in result.stdout.splitlines():
            if "Isaac" in line or "Omniverse" in line:
                window_id = line.split()[0]
                # Get window geometry
                geo_result = subprocess.run(
                    ["xdotool", "getwindowgeometry", "--shell", window_id],
                    capture_output=True,
                    text=True
                )
                geo = {}
                for geo_line in geo_result.stdout.splitlines():
                    if "=" in geo_line:
                        key, val = geo_line.split("=")
                        geo[key] = int(val)

                if all(k in geo for k in ["X", "Y", "WIDTH", "HEIGHT"]):
                    return (geo["X"], geo["Y"], geo["WIDTH"], geo["HEIGHT"])
    except Exception as e:
        print(f"Could not find Isaac Sim window: {e}")

    return None


def capture_screen_region(x: int, y: int, width: int, height: int, quality: int = 60) -> bytes:
    """Capture a screen region and return JPEG bytes."""
    if not HAS_MSS or not HAS_PIL:
        return b''

    with mss.mss() as sct:
        monitor = {"left": x, "top": y, "width": width, "height": height}
        screenshot = sct.grab(monitor)

        # Convert to PIL Image
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        # Resize for streaming efficiency (optional)
        max_width = 1280
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        # Convert to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def capture_full_screen(quality: int = 60) -> bytes:
    """Capture full screen and return JPEG bytes."""
    if not HAS_MSS or not HAS_PIL:
        return b''

    with mss.mss() as sct:
        # Capture primary monitor
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = sct.grab(monitor)

        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        # Resize for streaming
        max_width = 1280
        if img.width > max_width:
            ratio = max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((max_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        return buffer.getvalue()


def capture_worker(fps: int = 15, quality: int = 60):
    """Background worker that captures frames continuously."""
    global capture_running
    capture_running = True
    frame_time = 1.0 / fps

    # Try to find Isaac Sim window
    isaac_geo = get_isaac_window_geometry()

    while capture_running:
        start = time.time()

        try:
            if isaac_geo:
                x, y, w, h = isaac_geo
                frame = capture_screen_region(x, y, w, h, quality)
            else:
                frame = capture_full_screen(quality)

            # Put frame in queue (non-blocking, drop old frames)
            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(frame)
                except:
                    pass
        except Exception as e:
            print(f"Capture error: {e}")

        # Maintain frame rate
        elapsed = time.time() - start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)


async def generate_mjpeg() -> AsyncGenerator[bytes, None]:
    """Generate MJPEG stream from captured frames."""
    boundary = b"--frame\r\n"

    while True:
        try:
            # Get frame with timeout
            frame = await asyncio.get_event_loop().run_in_executor(
                None, lambda: frame_queue.get(timeout=1.0)
            )

            yield (
                boundary +
                b"Content-Type: image/jpeg\r\n" +
                f"Content-Length: {len(frame)}\r\n\r\n".encode() +
                frame +
                b"\r\n"
            )
        except queue.Empty:
            # No frame available, yield placeholder
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Stream error: {e}")
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    """Start the capture worker on app startup."""
    if HAS_MSS and HAS_PIL:
        thread = threading.Thread(target=capture_worker, daemon=True)
        thread.start()
        print("Screen capture worker started")
    else:
        print("WARNING: mss or PIL not installed. Install with: pip install mss pillow")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop capture on shutdown."""
    global capture_running
    capture_running = False


@app.get("/")
async def index():
    """Serve the streaming viewer page."""
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Isaac Sim Screen Stream</title>
    <style>
        body { margin: 0; padding: 20px; background: #1a1a2e; color: #e0e0e0; font-family: sans-serif; }
        h1 { color: #76ff03; }
        .container { max-width: 1400px; margin: 0 auto; }
        .stream-container { background: #000; border-radius: 10px; overflow: hidden; }
        img { max-width: 100%; height: auto; display: block; }
        .info { margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Isaac Sim 3D Stream</h1>
        <p>赤土崎多功能館 mmWave Fall Detection Demo</p>
        <div class="stream-container">
            <img src="/stream" alt="Isaac Sim Stream" />
        </div>
        <div class="info">
            <p><strong>說明:</strong> 此串流透過螢幕捕捉方式顯示 Isaac Sim 畫面</p>
            <p>MJPEG Stream URL: <code>/stream</code></p>
        </div>
    </div>
</body>
</html>
    """)


@app.get("/stream")
async def video_stream():
    """Stream MJPEG video."""
    if not HAS_MSS or not HAS_PIL:
        return Response(
            content="Screen capture not available. Install: pip install mss pillow",
            status_code=503
        )

    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/snapshot")
async def snapshot():
    """Get a single frame snapshot."""
    if not HAS_MSS or not HAS_PIL:
        return Response(
            content="Screen capture not available. Install: pip install mss pillow",
            status_code=503
        )

    frame = capture_full_screen(quality=80)
    return Response(content=frame, media_type="image/jpeg")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "mss_available": HAS_MSS,
        "pil_available": HAS_PIL,
        "capture_running": capture_running
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
