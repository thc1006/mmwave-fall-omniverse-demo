"""Record RTX Radar data for fall vs non-fall episodes.

This script is meant to be run inside the Isaac Sim Python environment, e.g.:

    ./python.sh sim/mmwave_fall_extension/record_fall_data.py --output-dir ml/data

It intentionally contains placeholders where you should wire in the official
RTX Radar examples (see Isaac Sim documentation for `rtx_radar.py`) and your
own animation logic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from isaacsim import SimulationApp  # type: ignore

# NOTE: `omni` imports must happen after SimulationApp is created.
simulation_app = SimulationApp({"headless": True})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record RTX Radar data for fall detection.")
    parser.add_argument("--output-dir", type=str, default="ml/data", help="Directory to store .npz episodes.")
    parser.add_argument("--episodes", type=int, default=10, help="Total episodes per class (fall / normal)."
    )
    parser.add_argument("--frames", type=int, default=128, help="Frames per episode.")
    return parser.parse_args()


def _bootstrap_scene():
    """Import Omniverse modules and set up the scene and radar sensor."""
    import omni.usd

    from . import scene, radar_sensor

    scene.setup_scene()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage was not created correctly.")

    # At this point you should also ensure that the RTX Radar is hooked into an
    # output buffer via an Action Graph or writer node (see Isaac Sim docs).


def _step_simulation(num_frames: int) -> np.ndarray:
    """Advance the simulation and collect radar frames.

    This is a **placeholder** implementation. In a real setup you would:

    - Use Isaac Sim's RTX Radar API to read from the RtxSensorCpu buffer every frame
    - Optionally convert the raw buffer to range-Doppler / point clouds
    - Stack frames into a [frames, features] numpy array
    """
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    dummy_dim = 256
    frames = []
    for _ in range(num_frames):
        simulation_app.update()
        # TODO: replace this with real radar buffer extraction.
        # Example shape: (dummy_dim,) or (range_bins * doppler_bins,)
        frames.append(np.random.randn(dummy_dim).astype("float32"))

    timeline.stop()
    return np.stack(frames, axis=0)


def _record_class_episodes(label: str, count: int, frames: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        data = _step_simulation(frames)
        path = out_dir / f"{label}_{idx:03d}.npz"
        np.savez_compressed(path, data=data, label=label)
        print(f"[record_fall_data] Saved {label} episode {idx} to {path}")


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    _bootstrap_scene()

    # Record normal (non-fall) motion episodes.
    _record_class_episodes("normal", args.episodes, args.frames, out_root / "normal")

    # TODO: Call an animation routine that includes a fall (跌倒) motion.
    # For now this uses the same dummy frames as normal; replace with real animation.
    _record_class_episodes("fall", args.episodes, args.frames, out_root / "fall")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
