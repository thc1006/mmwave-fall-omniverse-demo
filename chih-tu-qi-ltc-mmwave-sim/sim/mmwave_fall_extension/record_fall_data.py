#!/usr/bin/env python3
"""Record fall detection data from Isaac Sim RTX Radar.

This script runs inside Isaac Sim to:
1. Load the 赤土崎多功能館 USD scene
2. Spawn avatar(s) with animation controllers
3. Record radar data during fall and non-fall sequences
4. Save labeled data to ml/data/ for training

Usage (inside Isaac Sim):
    /isaac-sim/python.sh sim/mmwave_fall_extension/record_fall_data.py \
        --output-dir ml/data \
        --episodes 100 \
        --frames 128 \
        --scenario fall_incident \
        --stage sim/usd/chih_tu_qi_floor1_ltc.usd \
        --config facility/chih_tu_qi_floor1_ltc.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Try to import Isaac Sim modules
try:
    from omni.isaac.kit import SimulationApp

    # Must create SimulationApp before importing other omni modules
    CONFIG = {
        "renderer": "RayTracedLighting",
        "headless": True,
        "width": 1280,
        "height": 720,
    }
    simulation_app = SimulationApp(CONFIG)

    import omni
    import omni.usd
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import open_stage
    from pxr import Usd

    ISAAC_SIM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Isaac Sim not available: {e}")
    ISAAC_SIM_AVAILABLE = False
    simulation_app = None

# Import radar sensor module
try:
    from .radar_sensor import RadarConfig, RadarFrame, RadarSensorManager, load_radars_from_yaml
except ImportError:
    # Running as standalone script
    sys.path.insert(0, str(Path(__file__).parent))
    from radar_sensor import RadarConfig, RadarFrame, RadarSensorManager, load_radars_from_yaml


# =============================================================================
# Scenario Definitions
# =============================================================================

SCENARIOS = {
    "normal_operation": {
        "label": 0,
        "label_name": "normal",
        "description": "正常營運 - 站立、行走、坐下",
        "motion_types": ["standing", "walking", "sitting"],
        "duration_range": (3.0, 10.0),  # seconds
    },
    "fall_incident": {
        "label": 1,
        "label_name": "fall",
        "description": "跌倒事件 - 突然跌倒、緩慢倒下",
        "motion_types": ["sudden_fall", "slow_collapse", "trip_fall"],
        "duration_range": (1.0, 3.0),
    },
    "rehab_bad_posture": {
        "label": 2,
        "label_name": "rehab_bad_posture",
        "description": "復健姿勢不良 - 動作不正確",
        "motion_types": ["incorrect_stretch", "unbalanced_exercise"],
        "duration_range": (2.0, 5.0),
    },
    "chest_abnormal": {
        "label": 3,
        "label_name": "chest_abnormal",
        "description": "胸腔異常 - 呼吸急促、胸悶",
        "motion_types": ["rapid_breathing", "chest_pain_posture"],
        "duration_range": (2.0, 8.0),
    },
}


@dataclass
class RecordingConfig:
    """Configuration for data recording session."""

    output_dir: Path = Path("ml/data")
    episodes_per_label: int = 100
    frames_per_episode: int = 128
    frame_rate: float = 20.0  # Hz
    stage_path: str = "sim/usd/chih_tu_qi_floor1_ltc.usd"
    config_path: str = "facility/chih_tu_qi_floor1_ltc.yaml"
    scenario: str = "normal_operation"

    @property
    def frame_dt(self) -> float:
        """Time step between frames."""
        return 1.0 / self.frame_rate


class FallDataRecorder:
    """Records fall detection data from Isaac Sim simulation."""

    def __init__(self, config: RecordingConfig):
        """Initialize the recorder.

        Args:
            config: Recording configuration
        """
        self.config = config
        self.radar_manager = RadarSensorManager()
        self.world: World | None = None
        self._frame_buffer: list[np.ndarray] = []
        self._current_episode = 0

    def setup(self) -> bool:
        """Set up the simulation environment.

        Returns:
            True if setup was successful
        """
        if not ISAAC_SIM_AVAILABLE:
            logger.warning("Isaac Sim not available. Using synthetic data mode.")
            return self._setup_synthetic()

        try:
            # Open USD stage
            stage_path = Path(self.config.stage_path)
            if not stage_path.exists():
                logger.error(f"Stage not found: {stage_path}")
                return False

            open_stage(str(stage_path))
            logger.info(f"Opened stage: {stage_path}")

            # Initialize World
            self.world = World(stage_units_in_meters=1.0)
            self.world.initialize_physics()
            logger.info("World initialized")

            # Load radar configurations from YAML
            config_path = Path(self.config.config_path)
            if config_path.exists():
                radar_configs = load_radars_from_yaml(config_path)
                for rc in radar_configs:
                    self.radar_manager.add_radar(rc)
                logger.info(f"Loaded {len(radar_configs)} radar configurations")
            else:
                # Add default radar
                self.radar_manager.add_radar(RadarConfig())
                logger.warning("Using default radar configuration")

            return True

        except Exception as e:
            logger.exception(f"Setup failed: {e}")
            return False

    def _setup_synthetic(self) -> bool:
        """Set up synthetic data generation mode.

        Returns:
            True (always succeeds)
        """
        # Load radar configs if available
        config_path = Path(self.config.config_path)
        if config_path.exists():
            radar_configs = load_radars_from_yaml(config_path)
            for rc in radar_configs:
                self.radar_manager.configs[rc.radar_id] = rc
            logger.info(f"Loaded {len(radar_configs)} radar configurations (synthetic mode)")
        else:
            self.radar_manager.configs["radar_001"] = RadarConfig()

        return True

    def record_episode(self, scenario: str) -> tuple[np.ndarray, int]:
        """Record a single episode of radar data.

        Args:
            scenario: Scenario name from SCENARIOS

        Returns:
            Tuple of (data array [frames, features], label)
        """
        scenario_info = SCENARIOS.get(scenario)
        if scenario_info is None:
            logger.error(f"Unknown scenario: {scenario}")
            return np.zeros((self.config.frames_per_episode, 256)), 0

        label = scenario_info["label"]
        self._frame_buffer.clear()

        # Select random radar
        radar_ids = list(self.radar_manager.configs.keys())
        radar_id = np.random.choice(radar_ids) if radar_ids else "radar_001"

        # Record frames
        for frame_idx in range(self.config.frames_per_episode):
            timestamp = frame_idx * self.config.frame_dt

            if ISAAC_SIM_AVAILABLE and self.world is not None:
                # Step simulation
                self.world.step(render=True)

                # Get radar frame
                frame = self.radar_manager.get_frame(radar_id, timestamp)
            else:
                # Generate synthetic frame
                frame = self.radar_manager._generate_synthetic_frame(
                    radar_id, timestamp, scenario_info["label_name"]
                )

            # Convert to feature vector
            features = frame.to_feature_vector(flatten=True)
            self._frame_buffer.append(features)

        # Stack frames
        data = np.stack(self._frame_buffer, axis=0)
        return data, label

    def record_dataset(self) -> None:
        """Record full dataset for configured scenario."""
        scenario = self.config.scenario
        scenario_info = SCENARIOS.get(scenario)

        if scenario_info is None:
            logger.error(f"Unknown scenario: {scenario}")
            return

        # Create output directory
        output_dir = self.config.output_dir / scenario_info["label_name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Recording {self.config.episodes_per_label} episodes for '{scenario}'")
        logger.info(f"Output directory: {output_dir}")

        for episode in range(self.config.episodes_per_label):
            self._current_episode = episode

            # Record episode
            data, label = self.record_episode(scenario)

            # Save to file
            filename = f"episode_{episode:04d}.npz"
            filepath = output_dir / filename

            np.savez_compressed(
                filepath,
                data=data,
                label=label,
                label_name=scenario_info["label_name"],
                scenario=scenario,
                frames=self.config.frames_per_episode,
                frame_rate=self.config.frame_rate,
            )

            if (episode + 1) % 10 == 0:
                logger.info(f"  Recorded episode {episode + 1}/{self.config.episodes_per_label}")

        logger.info(f"Dataset recording complete: {self.config.episodes_per_label} episodes")

    def record_all_scenarios(self) -> None:
        """Record data for all scenarios."""
        for scenario in SCENARIOS:
            logger.info(f"\n{'='*60}")
            logger.info(f"Recording scenario: {scenario}")
            logger.info(f"{'='*60}")

            self.config.scenario = scenario
            self.record_dataset()

    def cleanup(self) -> None:
        """Clean up simulation resources."""
        if ISAAC_SIM_AVAILABLE and simulation_app is not None:
            simulation_app.close()
            logger.info("Simulation closed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record fall detection data from Isaac Sim."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/data"),
        help="Output directory for recorded data",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per label",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=128,
        help="Number of frames per episode",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all"] + list(SCENARIOS.keys()),
        help="Scenario to record (or 'all' for all scenarios)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="sim/usd/chih_tu_qi_floor1_ltc.usd",
        help="USD stage file path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="facility/chih_tu_qi_floor1_ltc.yaml",
        help="Facility configuration YAML path",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=20.0,
        help="Recording frame rate in Hz",
    )

    args = parser.parse_args()

    # Create recording configuration
    config = RecordingConfig(
        output_dir=args.output_dir,
        episodes_per_label=args.episodes,
        frames_per_episode=args.frames,
        frame_rate=args.frame_rate,
        stage_path=args.stage,
        config_path=args.config,
        scenario=args.scenario,
    )

    logger.info("="*60)
    logger.info("赤土崎多功能館 mmWave Fall Detection - Data Recording")
    logger.info("="*60)
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Episodes per label: {config.episodes_per_label}")
    logger.info(f"Frames per episode: {config.frames_per_episode}")
    logger.info(f"Frame rate: {config.frame_rate} Hz")
    logger.info(f"Scenario: {config.scenario}")
    logger.info("")

    # Initialize recorder
    recorder = FallDataRecorder(config)

    try:
        # Setup
        if not recorder.setup():
            logger.error("Setup failed. Exiting.")
            return

        # Record data
        if args.scenario == "all":
            recorder.record_all_scenarios()
        else:
            recorder.record_dataset()

        logger.info("\nData recording completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nRecording interrupted by user.")
    except Exception as e:
        logger.exception(f"Recording failed: {e}")
    finally:
        recorder.cleanup()


if __name__ == "__main__":
    main()
