"""Record RTX Radar data for multiple motion scenarios.

This script is meant to be run inside the Isaac Sim Python environment, e.g.:

    ./python.sh sim/mmwave_fall_extension/record_fall_data.py --output-dir ml/data

It currently uses a dummy random radar signal for demonstration purposes; you
should replace `_step_simulation` with code that reads from the RTX Radar CPU
buffer as shown in NVIDIA's RTX Radar examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from isaacsim import SimulationApp  # type: ignore

simulation_app = SimulationApp({"headless": True})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record RTX Radar data for fall detection scenarios.")
    parser.add_argument("--output-dir", type=str, default="ml/data", help="Directory to store .npz episodes.")
    parser.add_argument("--episodes", type=int, default=10, help="Total episodes per class.")
    parser.add_argument("--frames", type=int, default=128, help="Frames per episode.")
    return parser.parse_args()


def _bootstrap_scene():
    """Import Omniverse modules and set up the scene and radar sensor(s)."""
    import omni.usd

    from . import scene

    scene.setup_scene()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage was not created correctly.")

    # At this point you should also ensure that the RTX Radar is hooked into an
    # output buffer via an Action Graph or writer node (see Isaac Sim docs).


def _step_simulation(num_frames: int, scenario: str) -> np.ndarray:
    """Advance the simulation and collect radar frames.

    Parameters
    ----------
    num_frames:
        Number of frames to simulate.
    scenario:
        Motion scenario label, e.g. "normal", "fall", "rehab_bad_posture", "chest_abnormal".

    In a real setup you would:
    - Drive the animation using `scene.animate_fall_sequence(loop_count=1, scenario=scenario)`
    - Use Isaac Sim's RTX Radar API to read from the RtxSensorCpu buffer every frame
    - Convert the raw buffer to range-Doppler / point clouds / features

    For now this uses a dummy Gaussian random vector per frame as a placeholder.
    """
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    dummy_dim = 256
    frames = []
    for _ in range(num_frames):
        simulation_app.update()
        frames.append(np.random.randn(dummy_dim).astype("float32"))

    timeline.stop()
    return np.stack(frames, axis=0)


def _record_class_episodes(label: str, count: int, frames: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        data = _step_simulation(frames, scenario=label)
        path = out_dir / f"{label}_{idx:03d}.npz"
        np.savez_compressed(path, data=data, label=label)
        print(f"[record_fall_data] Saved {label} episode {idx} to {path}")


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    _bootstrap_scene()

    scenarios = ["normal", "fall", "rehab_bad_posture", "chest_abnormal"]
    for label in scenarios:
        _record_class_episodes(label, args.episodes, args.frames, out_root / label)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
