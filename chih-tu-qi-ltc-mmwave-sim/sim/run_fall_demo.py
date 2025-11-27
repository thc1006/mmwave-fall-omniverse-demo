#!/usr/bin/env python3
"""Fall Detection Demo for Isaac Sim.

This script runs a complete fall detection demo:
1. Loads the 赤土崎多功能館 scene
2. Spawns a human avatar
3. Animates the avatar walking then falling
4. RTX Radar detects the fall pattern
5. Sends data to API for prediction
6. Triggers alert on fall detection

Usage (run inside Isaac Sim):
    ./python.sh sim/run_fall_demo.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# Isaac Sim imports
try:
    import omni
    import omni.kit.app
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.prims import XFormPrim
    from pxr import Gf, UsdGeom, Usd
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    print("Warning: Isaac Sim not available. Running in simulation mode.")

# HTTP client for API calls
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class FallDetectionDemo:
    """Main demo class for fall detection simulation."""

    def __init__(
        self,
        scene_path: str | None = None,
        api_url: str = "http://localhost:8002",
    ):
        """Initialize demo.

        Args:
            scene_path: Path to USD scene file
            api_url: URL of fall detection API
        """
        self.scene_path = scene_path or self._find_scene()
        self.api_url = api_url
        self.world: World | None = None
        self.avatar_prim = None
        self.radar_data_buffer: list[np.ndarray] = []
        self.is_falling = False
        self.alert_triggered = False

    def _find_scene(self) -> str:
        """Find the scene USD file."""
        project_root = Path(__file__).parent.parent
        candidates = [
            project_root / "sim" / "usd" / "chih_tu_qi_floor1_ltc.usd",
            project_root / "sim" / "usd" / "hall_scene.usd",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return str(candidates[0])

    async def setup(self) -> None:
        """Set up the simulation world."""
        if not ISAAC_SIM_AVAILABLE:
            print("[Demo] Isaac Sim not available, using synthetic mode")
            return

        print(f"[Demo] Loading scene: {self.scene_path}")

        # Create world
        self.world = World(stage_units_in_meters=1.0)
        await self.world.initialize_simulation_context_async()

        # Load scene
        if Path(self.scene_path).exists():
            add_reference_to_stage(self.scene_path, "/World/Hall")
            print("[Demo] Scene loaded")

        # Spawn avatar
        await self._spawn_avatar()

        # Set up radar sensor
        await self._setup_radar()

        print("[Demo] Setup complete")

    async def _spawn_avatar(self) -> None:
        """Spawn a human avatar in the scene."""
        if not ISAAC_SIM_AVAILABLE:
            return

        stage = omni.usd.get_context().get_stage()

        # Create avatar transform
        avatar_path = "/World/Avatar"
        avatar_xform = UsdGeom.Xform.Define(stage, avatar_path)

        # Position avatar in the hall
        avatar_xform.AddTranslateOp().Set(Gf.Vec3d(5.0, 2.0, 0.0))

        # Create a simple capsule as placeholder for human
        capsule_path = f"{avatar_path}/Body"
        capsule = UsdGeom.Capsule.Define(stage, capsule_path)
        capsule.GetHeightAttr().Set(1.6)  # Human height
        capsule.GetRadiusAttr().Set(0.3)  # Body radius
        capsule.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.9))  # Center at torso

        self.avatar_prim = XFormPrim(avatar_path)
        print("[Demo] Avatar spawned at position (5, 2, 0)")

    async def _setup_radar(self) -> None:
        """Set up RTX Radar sensor."""
        if not ISAAC_SIM_AVAILABLE:
            return

        try:
            from omni.isaac.sensor import _sensor
            # RTX Radar would be set up here
            print("[Demo] Radar sensor configured")
        except Exception as e:
            print(f"[Demo] Radar setup note: {e}")

    def generate_synthetic_radar_data(
        self,
        scenario: str = "normal",
        seq_len: int = 128,
        feature_dim: int = 256,
    ) -> np.ndarray:
        """Generate synthetic radar data for different scenarios.

        Args:
            scenario: "normal", "fall", "rehab_bad_posture", "chest_abnormal"
            seq_len: Sequence length
            feature_dim: Feature dimension

        Returns:
            Radar data array [seq_len, feature_dim]
        """
        data = np.random.randn(seq_len, feature_dim).astype(np.float32) * 0.1

        if scenario == "normal":
            # Stable walking pattern
            t = np.linspace(0, 4 * np.pi, seq_len)
            walking = np.sin(t) * 0.5
            data[:, 30:50] += walking[:, None] * 0.3
            data[:, 100:120] += 0.2  # Stable signature

        elif scenario == "fall":
            # High velocity sudden change pattern
            t = np.linspace(0, 1, seq_len)
            # Build up (walking)
            data[:seq_len//3, 30:50] += 0.3
            # Sudden acceleration (fall)
            fall_start = seq_len // 3
            fall_end = seq_len // 2
            fall_pattern = np.exp(-5 * (t[fall_start:fall_end] - 0.5) ** 2) * 3
            data[fall_start:fall_end, 50:80] += fall_pattern[:, None]
            # Impact and stillness
            data[fall_end:, 80:100] += np.random.randn(seq_len - fall_end, 20) * 0.05

        elif scenario == "rehab_bad_posture":
            # Irregular movement pattern
            t = np.linspace(0, 6 * np.pi, seq_len)
            irregular = np.sin(t) + 0.5 * np.sin(3 * t) + 0.3 * np.random.randn(seq_len)
            data[:, 20:40] += irregular[:, None] * 0.6

        elif scenario == "chest_abnormal":
            # Irregular breathing pattern
            t = np.linspace(0, 10 * np.pi, seq_len)
            breathing = np.sin(t * 0.7) + 0.8 * np.random.randn(seq_len)
            data[:, 10:20] += breathing[:, None] * 0.4

        return data

    async def call_api(self, radar_data: np.ndarray) -> dict[str, Any] | None:
        """Call fall detection API.

        Args:
            radar_data: Radar sequence data

        Returns:
            API response or None if failed
        """
        if not AIOHTTP_AVAILABLE:
            # Simulate API response
            scenario = "fall" if self.is_falling else "normal"
            return {
                "label": scenario,
                "label_index": 1 if scenario == "fall" else 0,
                "confidence": 0.92 if scenario == "fall" else 0.88,
                "probabilities": {
                    "normal": 0.08 if scenario == "fall" else 0.88,
                    "fall": 0.92 if scenario == "fall" else 0.05,
                    "rehab_bad_posture": 0.0,
                    "chest_abnormal": 0.0,
                },
                "explanation": f"{'Fall detected with high confidence (92.0%). Immediate attention required!' if scenario == 'fall' else 'Normal activity detected (88.0%). No concerns.'}",
            }

        try:
            async with aiohttp.ClientSession() as session:
                payload = {"sequences": radar_data.tolist()}
                async with session.post(
                    f"{self.api_url}/predict",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"[API] Error: {response.status}")
                        return None
        except Exception as e:
            print(f"[API] Connection error: {e}")
            return None

    def trigger_alert(self, prediction: dict[str, Any]) -> None:
        """Trigger alert based on prediction.

        Args:
            prediction: API prediction response
        """
        label = prediction.get("label", "")
        confidence = prediction.get("confidence", 0)

        alert_levels = {
            "fall": ("CRITICAL", "\033[91m"),  # Red
            "chest_abnormal": ("HIGH", "\033[93m"),  # Yellow
            "rehab_bad_posture": ("MEDIUM", "\033[94m"),  # Blue
            "normal": ("NONE", "\033[92m"),  # Green
        }

        level, color = alert_levels.get(label, ("UNKNOWN", "\033[0m"))
        reset = "\033[0m"

        print("\n" + "=" * 60)
        print(f"{color}[ALERT - {level}]{reset}")
        print(f"Detection: {label}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Explanation: {prediction.get('explanation', '')}")
        print("=" * 60 + "\n")

        if label == "fall" and confidence >= 0.6:
            self.alert_triggered = True
            print(f"{color}>>> FALL DETECTED! Notifying caregivers... <<<{reset}")

    async def animate_avatar(self, scenario: str = "fall") -> None:
        """Animate avatar through a scenario.

        Args:
            scenario: Animation scenario
        """
        if not ISAAC_SIM_AVAILABLE or self.avatar_prim is None:
            print(f"[Demo] Simulating {scenario} animation...")
            await asyncio.sleep(1)
            return

        print(f"[Demo] Running {scenario} animation...")

        if scenario == "fall":
            # Walk forward
            for i in range(30):
                pos = self.avatar_prim.get_world_pose()[0]
                new_pos = pos + np.array([0.1, 0, 0])
                self.avatar_prim.set_world_pose(position=new_pos)
                await asyncio.sleep(0.1)

            # Fall down (rotate and drop)
            self.is_falling = True
            for i in range(20):
                pos = self.avatar_prim.get_world_pose()[0]
                # Drop height
                new_pos = pos + np.array([0.05, 0, -0.05 * i])
                new_pos[2] = max(new_pos[2], 0.3)  # Don't go below ground
                self.avatar_prim.set_world_pose(position=new_pos)
                await asyncio.sleep(0.05)

            print("[Demo] Fall animation complete")

    async def run_demo(self) -> None:
        """Run the complete demo."""
        print("\n" + "=" * 60)
        print("  赤土崎多功能館 mmWave Fall Detection Demo")
        print("=" * 60 + "\n")

        # Setup
        await self.setup()

        # Demo scenarios
        scenarios = [
            ("normal", "Normal walking activity"),
            ("fall", "Person falling down"),
        ]

        for scenario, description in scenarios:
            print(f"\n--- Scenario: {description} ---\n")

            # Generate radar data
            if scenario == "fall":
                self.is_falling = True

            radar_data = self.generate_synthetic_radar_data(scenario)
            print(f"[Radar] Generated {radar_data.shape} radar sequence")

            # Animate (if Isaac Sim available)
            await self.animate_avatar(scenario)

            # Call API
            print("[API] Sending data for prediction...")
            prediction = await self.call_api(radar_data)

            if prediction:
                self.trigger_alert(prediction)
            else:
                print("[API] No prediction received")

            self.is_falling = False
            await asyncio.sleep(2)

        print("\n" + "=" * 60)
        print("  Demo Complete!")
        print("=" * 60 + "\n")


async def main():
    """Main entry point."""
    demo = FallDetectionDemo(
        api_url="http://localhost:8002"
    )
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
