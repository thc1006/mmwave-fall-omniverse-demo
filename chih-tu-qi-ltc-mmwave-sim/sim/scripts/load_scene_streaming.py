#!/usr/bin/env python3
"""Load USD Scene for Isaac Sim WebRTC Streaming.

This script loads the Chih-Tu-Qi LTC hall scene (chih_tu_qi_floor1_ltc.usd)
into Isaac Sim and configures it for WebRTC streaming.

The script is designed to run inside the Isaac Sim container and will:
1. Initialize the simulation context
2. Load the USD stage
3. Configure viewport for streaming
4. Set up camera for optimal viewing
5. Keep the simulation running for streaming clients

Usage (inside Isaac Sim container):
    /isaac-sim/python.sh sim/scripts/load_scene_streaming.py

Environment variables:
    USD_STAGE_PATH: Path to USD file (default: sim/usd/chih_tu_qi_floor1_ltc.usd)
    CAMERA_POSITION: Camera position as "x,y,z" (default: "15,15,10")
    CAMERA_TARGET: Camera look-at target as "x,y,z" (default: "0,0,0")
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Determine project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def get_usd_stage_path() -> str:
    """Get the USD stage path from environment or default."""
    default_path = PROJECT_ROOT / "sim" / "usd" / "chih_tu_qi_floor1_ltc.usd"
    return os.environ.get("USD_STAGE_PATH", str(default_path))


def parse_vec3(value: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    """Parse a comma-separated vector string."""
    try:
        parts = [float(x.strip()) for x in value.split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    except (ValueError, AttributeError):
        pass
    return default


class SceneStreamingManager:
    """Manages USD scene loading and streaming configuration."""

    def __init__(
        self,
        usd_path: str,
        camera_position: tuple[float, float, float] = (15.0, 15.0, 10.0),
        camera_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """Initialize the streaming manager.

        Args:
            usd_path: Path to the USD stage file
            camera_position: Initial camera position (x, y, z)
            camera_target: Camera look-at target (x, y, z)
        """
        self.usd_path = usd_path
        self.camera_position = camera_position
        self.camera_target = camera_target
        self._running = True
        self._world = None

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown handlers."""
        def signal_handler(signum, frame):
            print(f"\n[Streaming] Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def initialize(self) -> bool:
        """Initialize Isaac Sim and load the scene.

        Returns:
            True if initialization succeeded, False otherwise
        """
        try:
            # Import Isaac Sim modules
            import carb
            import omni
            import omni.kit.app
            from omni.isaac.core import World
            from omni.isaac.core.utils.stage import add_reference_to_stage, open_stage_async
            from pxr import Gf, Usd, UsdGeom

            print("[Streaming] Isaac Sim modules loaded successfully")

            # Check if USD file exists
            if not Path(self.usd_path).exists():
                print(f"[Streaming] Warning: USD file not found: {self.usd_path}")
                print("[Streaming] Creating empty stage instead...")
                # Create a new stage if USD doesn't exist
                from omni.isaac.core.utils.stage import create_new_stage_async
                await create_new_stage_async()
            else:
                print(f"[Streaming] Loading USD stage: {self.usd_path}")
                # Open the existing stage
                result = await open_stage_async(self.usd_path)
                if not result:
                    print("[Streaming] Failed to open stage, creating new one")
                    from omni.isaac.core.utils.stage import create_new_stage_async
                    await create_new_stage_async()

            # Initialize the World
            self._world = World(stage_units_in_meters=1.0)
            await self._world.initialize_simulation_context_async()
            print("[Streaming] World initialized")

            # Set up camera for streaming
            await self._setup_camera()

            # Configure viewport for streaming
            await self._configure_viewport()

            print("[Streaming] Scene loaded and configured for streaming")
            return True

        except ImportError as e:
            print(f"[Streaming] Error: Failed to import Isaac Sim modules: {e}")
            print("[Streaming] Make sure you're running this inside Isaac Sim container")
            return False
        except Exception as e:
            print(f"[Streaming] Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _setup_camera(self) -> None:
        """Set up a camera for viewing the scene."""
        try:
            import omni
            from pxr import Gf, Usd, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print("[Streaming] Warning: No stage available for camera setup")
                return

            camera_path = "/World/StreamingCamera"

            # Check if camera already exists
            existing_prim = stage.GetPrimAtPath(camera_path)
            if existing_prim.IsValid():
                print(f"[Streaming] Using existing camera at {camera_path}")
                camera = UsdGeom.Camera(existing_prim)
            else:
                print(f"[Streaming] Creating camera at {camera_path}")
                camera = UsdGeom.Camera.Define(stage, camera_path)

            # Set camera properties
            camera.GetFocalLengthAttr().Set(24.0)
            camera.GetHorizontalApertureAttr().Set(36.0)
            camera.GetVerticalApertureAttr().Set(24.0)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

            # Position the camera
            xform = UsdGeom.Xformable(camera.GetPrim())
            xform.ClearXformOpOrder()

            # Calculate camera orientation (look-at)
            eye = Gf.Vec3d(*self.camera_position)
            target = Gf.Vec3d(*self.camera_target)
            up = Gf.Vec3d(0, 0, 1)

            # Create look-at matrix
            forward = (target - eye).GetNormalized()
            right = Gf.Cross(forward, up).GetNormalized()
            actual_up = Gf.Cross(right, forward)

            # Set translation
            translate_op = xform.AddTranslateOp()
            translate_op.Set(eye)

            # Set rotation (simplified - point camera at target)
            import math
            # Calculate angles
            dx = target[0] - eye[0]
            dy = target[1] - eye[1]
            dz = target[2] - eye[2]

            yaw = math.atan2(dy, dx) * 180 / math.pi
            pitch = math.atan2(-dz, math.sqrt(dx*dx + dy*dy)) * 180 / math.pi

            rotate_op = xform.AddRotateXYZOp()
            rotate_op.Set(Gf.Vec3f(pitch, 0, yaw + 90))

            print(f"[Streaming] Camera positioned at {self.camera_position}")
            print(f"[Streaming] Camera looking at {self.camera_target}")

            # Set as active camera for viewport
            try:
                import omni.kit.viewport.utility as viewport_utils
                viewport = viewport_utils.get_active_viewport()
                if viewport:
                    viewport.set_active_camera(camera_path)
                    print(f"[Streaming] Set active camera to {camera_path}")
            except Exception as e:
                print(f"[Streaming] Note: Could not set active camera: {e}")

        except Exception as e:
            print(f"[Streaming] Warning: Camera setup failed: {e}")

    async def _configure_viewport(self) -> None:
        """Configure viewport settings for optimal streaming."""
        try:
            import carb.settings

            settings = carb.settings.get_settings()

            # Streaming-optimized settings
            settings.set("/app/renderer/resolution/width", 1920)
            settings.set("/app/renderer/resolution/height", 1080)
            settings.set("/rtx/post/aa/op", 3)  # TAA
            settings.set("/rtx/directLighting/sampledLighting/enabled", True)
            settings.set("/app/asyncRendering", False)  # Sync for streaming

            # Viewport display settings
            settings.set("/persistent/app/viewport/displayOptions", 0)

            print("[Streaming] Viewport configured for streaming")

        except Exception as e:
            print(f"[Streaming] Warning: Viewport configuration failed: {e}")

    async def run(self) -> None:
        """Run the main streaming loop."""
        self._setup_signal_handlers()

        # Initialize scene
        if not await self.initialize():
            print("[Streaming] Failed to initialize, exiting")
            return

        print("\n" + "=" * 60)
        print("  Isaac Sim WebRTC Streaming Ready")
        print("=" * 60)
        print(f"  USD Stage: {self.usd_path}")
        print(f"  Access streaming at: http://localhost:8211/streaming/webrtc-demo")
        print("=" * 60 + "\n")

        # Main loop - keep simulation running for streaming
        try:
            import omni.kit.app

            while self._running:
                # Step the simulation
                if self._world is not None:
                    self._world.step(render=True)
                else:
                    # Fallback: just update the app
                    await omni.kit.app.get_app().next_update_async()

                # Small delay to prevent busy loop
                await asyncio.sleep(0.016)  # ~60 FPS

        except Exception as e:
            print(f"[Streaming] Error in main loop: {e}")
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        print("[Streaming] Cleaning up...")
        if self._world is not None:
            self._world.stop()
            self._world.clear()
        print("[Streaming] Cleanup complete")


async def main() -> int:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  Chih-Tu-Qi LTC Hall - WebRTC Streaming Loader")
    print("=" * 60 + "\n")

    # Parse configuration from environment
    usd_path = get_usd_stage_path()
    camera_pos = parse_vec3(
        os.environ.get("CAMERA_POSITION", "15,15,10"),
        (15.0, 15.0, 10.0)
    )
    camera_target = parse_vec3(
        os.environ.get("CAMERA_TARGET", "0,0,0"),
        (0.0, 0.0, 0.0)
    )

    print(f"[Config] USD Stage: {usd_path}")
    print(f"[Config] Camera Position: {camera_pos}")
    print(f"[Config] Camera Target: {camera_target}")
    print()

    # Create and run streaming manager
    manager = SceneStreamingManager(
        usd_path=usd_path,
        camera_position=camera_pos,
        camera_target=camera_target,
    )

    await manager.run()
    return 0


if __name__ == "__main__":
    # Check if we're running inside Isaac Sim
    try:
        import omni.kit.app
        # Running inside Isaac Sim - use its event loop
        import asyncio
        asyncio.ensure_future(main())
    except ImportError:
        # Not in Isaac Sim - run standalone (will fail but with clear message)
        print("=" * 60)
        print("ERROR: This script must be run inside Isaac Sim!")
        print("")
        print("Usage:")
        print("  /isaac-sim/python.sh sim/scripts/load_scene_streaming.py")
        print("")
        print("Or via Docker:")
        print("  docker exec isaac-streaming /isaac-sim/python.sh \\")
        print("    sim/scripts/load_scene_streaming.py")
        print("=" * 60)
        sys.exit(1)
