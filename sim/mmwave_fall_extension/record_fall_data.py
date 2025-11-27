"""Record RTX Radar data for multiple motion scenarios.

This script is meant to be run inside the Isaac Sim Python environment, e.g.:

    ./python.sh sim/mmwave_fall_extension/record_fall_data.py --output-dir ml/data

It uses Isaac Sim's RTX Radar API to read radar returns from the RtxSensorCpu buffer.
Falls back to dummy data if the radar buffer is unavailable.

References:
- RTX Radar Sensor Overview:
  https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html
- RtxSensorCpu API:
  https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.sensors.nv.radar/docs/index.html
- Radar Data Acquisition:
  https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#data-acquisition
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from isaacsim import SimulationApp  # type: ignore

simulation_app = SimulationApp({"headless": True})

# Import scipy for statistical feature computation (skew, kurtosis)
# Fallback to numpy-only computation if scipy is unavailable
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not available; skew and kurtosis will use numpy approximations.")


##############################################################################
# Constants and Configuration
##############################################################################

# Radar feature names extracted from RTX Radar returns
# Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#radar-point-cloud-output
RADAR_FEATURE_NAMES = [
    "range",       # Distance to target (meters)
    "velocity",    # Radial velocity (m/s, positive = approaching)
    "rcs",         # Radar Cross Section (dBsm)
    "snr",         # Signal-to-Noise Ratio (dB)
    "azimuth",     # Horizontal angle (radians)
    "elevation",   # Vertical angle (radians)
]

# Statistical aggregation functions applied to each feature per frame
STAT_NAMES = ["mean", "std", "min", "max", "skew", "kurtosis"]

# Total feature dimension per frame = len(RADAR_FEATURE_NAMES) * len(STAT_NAMES)
FEATURE_DIM = len(RADAR_FEATURE_NAMES) * len(STAT_NAMES)  # 6 * 6 = 36

# Dummy feature dimension used when radar data is unavailable
DUMMY_FEATURE_DIM = 256


##############################################################################
# Data Classes for Radar Returns
##############################################################################

@dataclass
class RadarReturn:
    """Single radar detection point from RTX Radar.

    Attributes mirror the RTX Radar point cloud output format.
    Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html
    """
    range_m: float           # Range in meters
    velocity_mps: float      # Radial velocity in m/s
    rcs_dbsm: float          # Radar cross section in dBsm
    snr_db: float            # Signal-to-noise ratio in dB
    azimuth_rad: float       # Azimuth angle in radians
    elevation_rad: float     # Elevation angle in radians


@dataclass
class RadarFrame:
    """Collection of radar returns for a single frame/scan."""
    timestamp: float
    returns: List[RadarReturn] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.returns)

    def to_array(self) -> np.ndarray:
        """Convert returns to numpy array of shape (N, 6)."""
        if not self.returns:
            return np.zeros((0, len(RADAR_FEATURE_NAMES)), dtype=np.float32)
        return np.array([
            [r.range_m, r.velocity_mps, r.rcs_dbsm, r.snr_db, r.azimuth_rad, r.elevation_rad]
            for r in self.returns
        ], dtype=np.float32)


##############################################################################
# Statistical Feature Extraction
##############################################################################

def compute_skew(arr: np.ndarray) -> float:
    """Compute skewness of array.

    Uses scipy.stats.skew if available, otherwise numpy approximation.
    """
    if len(arr) < 3:
        return 0.0
    if HAS_SCIPY:
        return float(scipy_stats.skew(arr, nan_policy="omit"))
    # Numpy approximation for skewness
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def compute_kurtosis(arr: np.ndarray) -> float:
    """Compute kurtosis of array (excess kurtosis, Fisher definition).

    Uses scipy.stats.kurtosis if available, otherwise numpy approximation.
    """
    if len(arr) < 4:
        return 0.0
    if HAS_SCIPY:
        return float(scipy_stats.kurtosis(arr, nan_policy="omit"))
    # Numpy approximation for excess kurtosis
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3.0)


def extract_statistical_features(radar_frame: RadarFrame) -> np.ndarray:
    """Extract statistical features from a radar frame.

    For each radar feature (range, velocity, rcs, snr, azimuth, elevation),
    compute: mean, std, min, max, skew, kurtosis.

    Returns:
        Feature vector of shape (FEATURE_DIM,) = (36,)

    Reference:
        Feature extraction for radar-based activity recognition:
        https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#applications
    """
    data = radar_frame.to_array()

    # Handle empty frames - return zeros
    if data.shape[0] == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    features = []
    for col_idx in range(data.shape[1]):
        col_data = data[:, col_idx]
        features.extend([
            np.mean(col_data),
            np.std(col_data),
            np.min(col_data),
            np.max(col_data),
            compute_skew(col_data),
            compute_kurtosis(col_data),
        ])

    return np.array(features, dtype=np.float32)


##############################################################################
# RTX Radar Buffer Interface
##############################################################################

class RtxRadarInterface:
    """Interface for reading RTX Radar data from Isaac Sim.

    This class wraps the RtxSensorCpu buffer access for the RTX Radar sensor.

    References:
    - Creating RTX Radar in Python:
      https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#python-scripting
    - Accessing radar data via annotator:
      https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#data-acquisition
    """

    def __init__(self, radar_prim_path: str = "/World/Radar"):
        """Initialize radar interface.

        Args:
            radar_prim_path: USD path to the RTX Radar prim
        """
        self.radar_prim_path = radar_prim_path
        self._annotator: Optional[Any] = None
        self._is_initialized = False
        self._use_dummy_data = False

    def initialize(self) -> bool:
        """Initialize the radar data annotator.

        Returns:
            True if real radar interface is available, False if falling back to dummy.
        """
        try:
            # Import Replicator for annotator access
            # Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_annotators.html
            import omni.replicator.core as rep

            # Create annotator for RTX Radar point cloud data
            # The "RtxSensorCpuIsaacRTXRadarPointCloud" annotator provides radar returns
            # Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#data-acquisition
            self._annotator = rep.AnnotatorRegistry.get_annotator(
                "RtxSensorCpuIsaacRTXRadarPointCloud"
            )
            self._annotator.attach([self.radar_prim_path])
            self._is_initialized = True
            print(f"[RtxRadarInterface] Successfully initialized radar at {self.radar_prim_path}")
            return True

        except Exception as e:
            warnings.warn(
                f"Failed to initialize RTX Radar interface: {e}. "
                "Falling back to dummy data generation."
            )
            self._use_dummy_data = True
            self._is_initialized = True
            return False

    def get_radar_frame(self, timestamp: float = 0.0) -> RadarFrame:
        """Read current radar frame from the sensor buffer.

        Args:
            timestamp: Current simulation time for the frame

        Returns:
            RadarFrame containing all radar detections

        Reference:
            Radar point cloud format:
            https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#radar-point-cloud-output
        """
        if not self._is_initialized:
            self.initialize()

        if self._use_dummy_data:
            return self._generate_dummy_frame(timestamp)

        try:
            # Get data from annotator
            # The annotator returns a structured array with radar point cloud data
            data = self._annotator.get_data()

            if data is None or len(data) == 0:
                return RadarFrame(timestamp=timestamp, returns=[])

            # Parse the radar point cloud data
            # RTX Radar output format includes: position, velocity, RCS, SNR, etc.
            # Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#output-format
            returns = []

            # Handle different data formats from the annotator
            if isinstance(data, dict):
                # Dictionary format with named fields
                positions = data.get("position", data.get("data", np.array([])))
                velocities = data.get("velocity", np.zeros_like(positions[:, 0]) if len(positions) > 0 else np.array([]))
                rcs_values = data.get("rcs", np.zeros_like(positions[:, 0]) if len(positions) > 0 else np.array([]))
                snr_values = data.get("snr", np.zeros_like(positions[:, 0]) if len(positions) > 0 else np.array([]))
                azimuth_values = data.get("azimuth", np.zeros_like(positions[:, 0]) if len(positions) > 0 else np.array([]))
                elevation_values = data.get("elevation", np.zeros_like(positions[:, 0]) if len(positions) > 0 else np.array([]))

                # Convert positions to range if needed
                if len(positions) > 0:
                    if positions.ndim == 2 and positions.shape[1] >= 3:
                        # positions is (N, 3) xyz coordinates - compute range
                        ranges = np.linalg.norm(positions, axis=1)
                    else:
                        ranges = positions.flatten()
                else:
                    ranges = np.array([])

            elif isinstance(data, np.ndarray):
                # Structured numpy array format
                if data.dtype.names is not None:
                    # Named fields in structured array
                    ranges = data["range"] if "range" in data.dtype.names else np.linalg.norm(data["position"], axis=1) if "position" in data.dtype.names else data[:, 0]
                    velocities = data["velocity"] if "velocity" in data.dtype.names else np.zeros(len(data))
                    rcs_values = data["rcs"] if "rcs" in data.dtype.names else np.zeros(len(data))
                    snr_values = data["snr"] if "snr" in data.dtype.names else np.zeros(len(data))
                    azimuth_values = data["azimuth"] if "azimuth" in data.dtype.names else np.zeros(len(data))
                    elevation_values = data["elevation"] if "elevation" in data.dtype.names else np.zeros(len(data))
                else:
                    # Plain array - assume columns are in order
                    num_cols = data.shape[1] if data.ndim == 2 else 1
                    if data.ndim == 1:
                        data = data.reshape(-1, 1)
                    ranges = data[:, 0] if num_cols > 0 else np.array([])
                    velocities = data[:, 1] if num_cols > 1 else np.zeros(len(data))
                    rcs_values = data[:, 2] if num_cols > 2 else np.zeros(len(data))
                    snr_values = data[:, 3] if num_cols > 3 else np.zeros(len(data))
                    azimuth_values = data[:, 4] if num_cols > 4 else np.zeros(len(data))
                    elevation_values = data[:, 5] if num_cols > 5 else np.zeros(len(data))
            else:
                warnings.warn(f"Unexpected radar data format: {type(data)}")
                return RadarFrame(timestamp=timestamp, returns=[])

            # Create RadarReturn objects
            for i in range(len(ranges)):
                returns.append(RadarReturn(
                    range_m=float(ranges[i]) if i < len(ranges) else 0.0,
                    velocity_mps=float(velocities[i]) if i < len(velocities) else 0.0,
                    rcs_dbsm=float(rcs_values[i]) if i < len(rcs_values) else 0.0,
                    snr_db=float(snr_values[i]) if i < len(snr_values) else 0.0,
                    azimuth_rad=float(azimuth_values[i]) if i < len(azimuth_values) else 0.0,
                    elevation_rad=float(elevation_values[i]) if i < len(elevation_values) else 0.0,
                ))

            return RadarFrame(timestamp=timestamp, returns=returns)

        except Exception as e:
            warnings.warn(f"Error reading radar data: {e}. Using dummy data for this frame.")
            return self._generate_dummy_frame(timestamp)

    def _generate_dummy_frame(self, timestamp: float) -> RadarFrame:
        """Generate dummy radar frame for testing when real sensor is unavailable.

        Generates realistic-looking radar returns for a person in a room.
        """
        # Simulate 10-50 radar returns (typical for a person in a room)
        num_returns = np.random.randint(10, 51)

        returns = []
        for _ in range(num_returns):
            returns.append(RadarReturn(
                range_m=np.random.uniform(1.0, 8.0),           # 1-8 meters typical indoor range
                velocity_mps=np.random.uniform(-2.0, 2.0),     # Walking/falling velocity
                rcs_dbsm=np.random.uniform(-20.0, 0.0),        # Human RCS typically -20 to 0 dBsm
                snr_db=np.random.uniform(10.0, 40.0),          # Typical SNR range
                azimuth_rad=np.random.uniform(-0.5, 0.5),      # ~±30 degrees FOV
                elevation_rad=np.random.uniform(-0.3, 0.3),    # ~±17 degrees elevation
            ))

        return RadarFrame(timestamp=timestamp, returns=returns)


# Global radar interface instance
_radar_interface: Optional[RtxRadarInterface] = None


def get_radar_interface(radar_prim_path: str = "/World/Radar") -> RtxRadarInterface:
    """Get or create the global radar interface instance."""
    global _radar_interface
    if _radar_interface is None:
        _radar_interface = RtxRadarInterface(radar_prim_path)
    return _radar_interface


##############################################################################
# Animation Control
##############################################################################

class AnimationController:
    """Control animation playback for different motion scenarios.

    Uses Omniverse Timeline API for animation control.

    Reference:
        Timeline API:
        https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/timeline.html
    """

    # Animation configuration per scenario
    # Maps scenario name to (animation_speed_multiplier, characteristic_motion)
    SCENARIO_CONFIG: Dict[str, Dict[str, Any]] = {
        "normal": {
            "speed": 1.0,
            "description": "Normal walking or standing motion",
            "velocity_range": (-0.5, 1.5),  # Typical walking velocity
        },
        "fall": {
            "speed": 1.5,
            "description": "Fall event - rapid downward motion",
            "velocity_range": (-3.0, 0.5),  # Rapid downward velocity during fall
        },
        "rehab_bad_posture": {
            "speed": 0.8,
            "description": "Rehabilitation exercise with poor posture",
            "velocity_range": (-0.3, 0.8),  # Slower, limited motion
        },
        "chest_abnormal": {
            "speed": 1.0,
            "description": "Abnormal chest/breathing pattern",
            "velocity_range": (-0.2, 0.2),  # Micro-motion from breathing
        },
    }

    def __init__(self):
        self._timeline = None

    def initialize(self):
        """Initialize timeline interface."""
        import omni.timeline
        self._timeline = omni.timeline.get_timeline_interface()

    def start_playback(self, scenario: str = "normal"):
        """Start animation playback for the specified scenario.

        Args:
            scenario: One of "normal", "fall", "rehab_bad_posture", "chest_abnormal"
        """
        if self._timeline is None:
            self.initialize()

        config = self.SCENARIO_CONFIG.get(scenario, self.SCENARIO_CONFIG["normal"])

        # Set playback speed based on scenario
        # Reference: https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/timeline.html#controlling-playback
        self._timeline.set_time_codes_per_second(24.0 * config["speed"])

        # Start playback from beginning
        self._timeline.set_current_time(0.0)
        self._timeline.play()

        print(f"[AnimationController] Started playback for scenario: {scenario} ({config['description']})")

    def stop_playback(self):
        """Stop animation playback."""
        if self._timeline is not None:
            self._timeline.stop()

    def get_current_time(self) -> float:
        """Get current timeline time in seconds."""
        if self._timeline is None:
            return 0.0
        return self._timeline.get_current_time()

    def is_playing(self) -> bool:
        """Check if timeline is currently playing."""
        if self._timeline is None:
            return False
        return self._timeline.is_playing()


##############################################################################
# Argument Parsing
##############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record RTX Radar data for fall detection scenarios.")
    parser.add_argument("--output-dir", type=str, default="ml/data", help="Directory to store .npz episodes.")
    parser.add_argument("--episodes", type=int, default=10, help="Total episodes per class.")
    parser.add_argument("--frames", type=int, default=128, help="Frames per episode.")
    parser.add_argument("--radar-path", type=str, default="/World/Radar", help="USD prim path to RTX Radar sensor.")
    parser.add_argument("--use-dummy", action="store_true", help="Force use of dummy data (for testing without Isaac Sim).")
    return parser.parse_args()


##############################################################################
# Scene Bootstrap and Simulation
##############################################################################

def _bootstrap_scene(radar_prim_path: str = "/World/Radar", use_dummy: bool = False) -> RtxRadarInterface:
    """Import Omniverse modules and set up the scene and radar sensor(s).

    Args:
        radar_prim_path: USD prim path to the RTX Radar sensor
        use_dummy: Force use of dummy data

    Returns:
        Initialized RtxRadarInterface instance

    Reference:
        Scene setup in Isaac Sim:
        https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html
    """
    import omni.usd

    from . import scene

    scene.setup_scene()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage was not created correctly.")

    # Initialize the RTX Radar interface
    # Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#python-scripting
    radar_interface = get_radar_interface(radar_prim_path)

    if use_dummy:
        radar_interface._use_dummy_data = True
        radar_interface._is_initialized = True
        print("[_bootstrap_scene] Using dummy radar data (--use-dummy flag set)")
    else:
        success = radar_interface.initialize()
        if not success:
            print("[_bootstrap_scene] RTX Radar not available, using dummy data")

    return radar_interface


def _step_simulation(
    num_frames: int,
    scenario: str,
    radar_interface: RtxRadarInterface,
    animation_controller: AnimationController,
) -> np.ndarray:
    """Advance the simulation and collect radar frames with feature extraction.

    Parameters
    ----------
    num_frames:
        Number of frames to simulate.
    scenario:
        Motion scenario label: "normal", "fall", "rehab_bad_posture", "chest_abnormal".
    radar_interface:
        Initialized RtxRadarInterface for reading radar data.
    animation_controller:
        AnimationController for timeline playback control.

    Returns
    -------
    np.ndarray
        Array of shape (num_frames, FEATURE_DIM) containing extracted features.
        Each row contains statistical features (mean, std, min, max, skew, kurtosis)
        for each radar parameter (range, velocity, rcs, snr, azimuth, elevation).

    Reference:
        RTX Radar data acquisition workflow:
        https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#data-acquisition
    """
    # Start animation playback for this scenario
    # Reference: https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/timeline.html
    animation_controller.start_playback(scenario)

    frames: List[np.ndarray] = []
    detection_counts: List[int] = []

    for frame_idx in range(num_frames):
        # Advance simulation by one step
        # Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/core_api_tutorials/tutorial_core_hello_world.html#simulation-step
        simulation_app.update()

        # Get current timestamp from animation controller
        timestamp = animation_controller.get_current_time()

        # Read radar data from RTX Radar buffer
        # Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#radar-point-cloud-output
        radar_frame = radar_interface.get_radar_frame(timestamp=timestamp)
        detection_counts.append(radar_frame.num_detections)

        # Extract statistical features from radar returns
        # Features: mean, std, min, max, skew, kurtosis for each of:
        # range, velocity, rcs, snr, azimuth, elevation
        features = extract_statistical_features(radar_frame)
        frames.append(features)

        # Log progress every 32 frames
        if (frame_idx + 1) % 32 == 0:
            avg_detections = np.mean(detection_counts[-32:])
            print(f"  [_step_simulation] Frame {frame_idx + 1}/{num_frames}, "
                  f"avg detections: {avg_detections:.1f}")

    # Stop animation playback
    animation_controller.stop_playback()

    # Stack all frames into a single array
    result = np.stack(frames, axis=0)

    # Log summary statistics
    total_avg_detections = np.mean(detection_counts)
    print(f"  [_step_simulation] Scenario '{scenario}' complete: "
          f"{num_frames} frames, avg {total_avg_detections:.1f} detections/frame, "
          f"feature shape: {result.shape}")

    return result


def _step_simulation_with_raw_data(
    num_frames: int,
    scenario: str,
    radar_interface: RtxRadarInterface,
    animation_controller: AnimationController,
) -> Dict[str, np.ndarray]:
    """Advanced version that returns both features and raw radar data.

    This is useful for debugging or if you want to train on raw point clouds
    rather than statistical features.

    Parameters
    ----------
    num_frames:
        Number of frames to simulate.
    scenario:
        Motion scenario label.
    radar_interface:
        Initialized RtxRadarInterface.
    animation_controller:
        AnimationController for timeline playback.

    Returns
    -------
    dict
        Dictionary containing:
        - "features": np.ndarray of shape (num_frames, FEATURE_DIM)
        - "raw_frames": List of np.ndarray, each of shape (N_i, 6) where N_i varies
        - "detection_counts": np.ndarray of shape (num_frames,)
        - "timestamps": np.ndarray of shape (num_frames,)

    Reference:
        For advanced use cases requiring raw radar data:
        https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#output-format
    """
    animation_controller.start_playback(scenario)

    features_list: List[np.ndarray] = []
    raw_frames_list: List[np.ndarray] = []
    detection_counts: List[int] = []
    timestamps: List[float] = []

    for frame_idx in range(num_frames):
        simulation_app.update()

        timestamp = animation_controller.get_current_time()
        timestamps.append(timestamp)

        radar_frame = radar_interface.get_radar_frame(timestamp=timestamp)
        detection_counts.append(radar_frame.num_detections)

        # Store raw data
        raw_data = radar_frame.to_array()
        raw_frames_list.append(raw_data)

        # Extract and store statistical features
        features = extract_statistical_features(radar_frame)
        features_list.append(features)

    animation_controller.stop_playback()

    return {
        "features": np.stack(features_list, axis=0),
        "raw_frames": raw_frames_list,
        "detection_counts": np.array(detection_counts, dtype=np.int32),
        "timestamps": np.array(timestamps, dtype=np.float64),
    }


def _record_class_episodes(
    label: str,
    count: int,
    frames: int,
    out_dir: Path,
    radar_interface: RtxRadarInterface,
    animation_controller: AnimationController,
    save_raw: bool = False,
):
    """Record multiple episodes for a single scenario class.

    Args:
        label: Scenario label (e.g., "fall", "normal")
        count: Number of episodes to record
        frames: Number of frames per episode
        out_dir: Output directory for .npz files
        radar_interface: Initialized radar interface
        animation_controller: Animation controller for playback
        save_raw: If True, also save raw radar data (larger files)

    Reference:
        Data storage format for ML training:
        https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[record_fall_data] Recording {count} episodes for scenario: {label}")
    print(f"  Output directory: {out_dir}")
    print(f"  Frames per episode: {frames}")
    print(f"  Feature dimension: {FEATURE_DIM}")

    for idx in range(count):
        print(f"\n[record_fall_data] Episode {idx + 1}/{count} for '{label}'...")

        if save_raw:
            # Save both features and raw data
            result = _step_simulation_with_raw_data(
                frames,
                scenario=label,
                radar_interface=radar_interface,
                animation_controller=animation_controller,
            )
            path = out_dir / f"{label}_{idx:03d}.npz"
            np.savez_compressed(
                path,
                data=result["features"],
                label=label,
                detection_counts=result["detection_counts"],
                timestamps=result["timestamps"],
                # Note: raw_frames is a list of variable-length arrays, stored as object array
                raw_frames=np.array(result["raw_frames"], dtype=object),
                feature_names=np.array(RADAR_FEATURE_NAMES),
                stat_names=np.array(STAT_NAMES),
            )
        else:
            # Save only statistical features (smaller files, recommended for training)
            data = _step_simulation(
                frames,
                scenario=label,
                radar_interface=radar_interface,
                animation_controller=animation_controller,
            )
            path = out_dir / f"{label}_{idx:03d}.npz"
            np.savez_compressed(
                path,
                data=data,
                label=label,
                feature_names=np.array(RADAR_FEATURE_NAMES),
                stat_names=np.array(STAT_NAMES),
            )

        print(f"[record_fall_data] Saved {label} episode {idx} to {path}")


##############################################################################
# Main Entry Point
##############################################################################

def main():
    """Main entry point for radar data recording.

    Supported scenarios:
    - "normal": Normal walking or standing motion
    - "fall": Fall event with rapid downward motion
    - "rehab_bad_posture": Rehabilitation exercise with poor posture
    - "chest_abnormal": Abnormal chest/breathing pattern

    Example usage:
        ./python.sh sim/mmwave_fall_extension/record_fall_data.py \\
            --output-dir ml/data \\
            --episodes 20 \\
            --frames 128 \\
            --radar-path /World/Radar

    Reference:
        Running scripts in Isaac Sim:
        https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html
    """
    args = parse_args()
    out_root = Path(args.output_dir)

    print("=" * 70)
    print("RTX Radar Data Recording for Fall Detection")
    print("=" * 70)
    print(f"Output directory: {out_root}")
    print(f"Episodes per scenario: {args.episodes}")
    print(f"Frames per episode: {args.frames}")
    print(f"Radar prim path: {args.radar_path}")
    print(f"Use dummy data: {args.use_dummy}")
    print(f"Feature dimension: {FEATURE_DIM} (6 features x 6 statistics)")
    print("=" * 70)

    # Initialize scene and radar interface
    radar_interface = _bootstrap_scene(
        radar_prim_path=args.radar_path,
        use_dummy=args.use_dummy,
    )

    # Initialize animation controller
    animation_controller = AnimationController()
    animation_controller.initialize()

    # Supported scenarios for fall detection
    # Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_radar.html#applications
    scenarios = ["normal", "fall", "rehab_bad_posture", "chest_abnormal"]

    print(f"\nRecording scenarios: {scenarios}")
    print(f"Total episodes to record: {len(scenarios) * args.episodes}")
    print()

    for label in scenarios:
        _record_class_episodes(
            label=label,
            count=args.episodes,
            frames=args.frames,
            out_dir=out_root / label,
            radar_interface=radar_interface,
            animation_controller=animation_controller,
            save_raw=False,  # Set to True to save raw point cloud data
        )

    print("\n" + "=" * 70)
    print("Recording complete!")
    print(f"Data saved to: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
