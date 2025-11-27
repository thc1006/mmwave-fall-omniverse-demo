#!/usr/bin/env python3
"""RTX Radar sensor integration for mmWave fall detection.

This module provides complete wrappers around Isaac Sim's RTX Radar sensor APIs
to simulate mmWave radar sensing in the 赤土崎多功能館 scene.

Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_based_radar.html
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Omniverse/Isaac Sim modules
try:
    import carb
    import omni.kit.commands
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom

    OMNIVERSE_AVAILABLE = True
except ImportError:
    OMNIVERSE_AVAILABLE = False
    logger.warning("Omniverse/Isaac Sim not available. Running in standalone mode.")

# Try to import RTX Radar sensor
try:
    from isaacsim.sensors.rtx import RtxRadar
    ISAACSIM_SENSORS_AVAILABLE = True
except ImportError:
    try:
        from omni.isaac.sensor import RtxRadar
        ISAACSIM_SENSORS_AVAILABLE = True
    except ImportError:
        ISAACSIM_SENSORS_AVAILABLE = False
        logger.warning("RTX Radar sensor not available")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RadarConfig:
    """Configuration for RTX Radar sensor.

    Mimics TI IWR6843ISK mmWave radar specifications.
    """

    # Radar identification
    radar_id: str = "radar_001"
    name: str = "mmWave Radar"

    # Position and orientation
    position: Tuple[float, float, float] = (0.0, 2.5, 0.0)  # x, y, z in meters
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll in degrees

    # Radar parameters (mimicking TI IWR6843ISK)
    fov_horizontal: float = 120.0  # degrees
    fov_vertical: float = 60.0     # degrees
    range_max: float = 10.0        # meters
    range_min: float = 0.1         # meters
    range_resolution: float = 0.04  # meters (~4cm)
    velocity_max: float = 10.0     # m/s
    velocity_resolution: float = 0.1  # m/s

    # Sensor output configuration
    num_range_bins: int = 256
    num_doppler_bins: int = 64
    num_azimuth_bins: int = 64

    # Update rate
    update_rate_hz: float = 20.0

    # Coverage and alert
    coverage_zones: List[str] = field(default_factory=list)
    alert_priority: str = "medium"


@dataclass
class RadarFrame:
    """Single frame of radar data."""

    timestamp: float
    radar_id: str

    # Point cloud data [N, 4] - x, y, z, velocity
    point_cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))

    # Range-Doppler map [range_bins, doppler_bins]
    range_doppler_map: np.ndarray = field(default_factory=lambda: np.zeros((256, 64)))

    # Range-Azimuth map [range_bins, azimuth_bins]
    range_azimuth_map: np.ndarray = field(default_factory=lambda: np.zeros((256, 64)))

    # Detected targets [N, 5] - x, y, z, velocity, rcs
    targets: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))

    # Metadata
    num_detections: int = 0
    zone_id: str = ""

    def to_feature_vector(self, flatten: bool = True) -> np.ndarray:
        """Convert radar frame to feature vector for ML model.

        Args:
            flatten: If True, return flattened 256-dim vector

        Returns:
            Feature vector of shape [256] if flatten, else raw maps
        """
        if flatten:
            # Subsample range-doppler map to 16x16 = 256 features
            rd_subsampled = self.range_doppler_map[::16, ::4]  # [16, 16]
            return rd_subsampled.flatten().astype(np.float32)

        # Return stacked maps
        return np.stack([
            self.range_doppler_map,
            self.range_azimuth_map,
        ], axis=0).astype(np.float32)

    def to_statistical_features(self) -> np.ndarray:
        """Extract statistical features from radar frame.

        Returns 36-dimensional feature vector:
        - 6 features (range, velocity, rcs, snr, azimuth, elevation)
        - 6 statistics each (mean, std, min, max, skew, kurtosis)
        """
        if self.num_detections == 0:
            return np.zeros(36, dtype=np.float32)

        # Use point cloud for feature extraction
        if self.point_cloud.shape[0] > 0:
            features = []
            for col in range(min(6, self.point_cloud.shape[1])):
                col_data = self.point_cloud[:, col]
                features.extend([
                    np.mean(col_data),
                    np.std(col_data),
                    np.min(col_data),
                    np.max(col_data),
                    _compute_skew(col_data),
                    _compute_kurtosis(col_data),
                ])
            # Pad if needed
            while len(features) < 36:
                features.extend([0.0] * 6)
            return np.array(features[:36], dtype=np.float32)

        return np.zeros(36, dtype=np.float32)


def _compute_skew(arr: np.ndarray) -> float:
    """Compute skewness of array."""
    if len(arr) < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _compute_kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis of array."""
    if len(arr) < 4:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-9:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4) - 3.0)


# =============================================================================
# Radar Sensor Manager
# =============================================================================


class RadarSensorManager:
    """Manager for RTX Radar sensors in Isaac Sim scene."""

    def __init__(self, world: Optional[Any] = None):
        """Initialize radar sensor manager.

        Args:
            world: Isaac Sim World instance (optional)
        """
        self.world = world
        self.radars: Dict[str, Any] = {}
        self.configs: Dict[str, RadarConfig] = {}
        self._stage: Optional[Any] = None

    @property
    def stage(self) -> Optional[Any]:
        """Get current USD stage."""
        if self._stage is not None:
            return self._stage
        if OMNIVERSE_AVAILABLE:
            try:
                self._stage = omni.usd.get_context().get_stage()
            except Exception:
                pass
        return self._stage

    def add_radar(self, config: RadarConfig) -> bool:
        """Add a radar sensor to the scene.

        Args:
            config: Radar configuration

        Returns:
            True if radar was added successfully
        """
        self.configs[config.radar_id] = config

        if not ISAACSIM_SENSORS_AVAILABLE:
            logger.warning(f"RTX Radar not available, using synthetic data for {config.radar_id}")
            return False

        try:
            # Create radar prim path
            radar_path = f"/World/Radars/{config.radar_id}"

            # Create RTX Radar sensor
            radar = RtxRadar(
                prim_path=radar_path,
                name=config.name,
                position=np.array(config.position),
                orientation=_euler_to_quat(config.rotation),
            )

            # Configure radar parameters
            radar.set_fov(config.fov_horizontal, config.fov_vertical)
            radar.set_range(config.range_min, config.range_max)
            radar.set_resolution(config.range_resolution, config.velocity_resolution)

            self.radars[config.radar_id] = radar
            logger.info(f"Added radar: {config.radar_id} at {config.position}")
            return True

        except Exception as e:
            logger.error(f"Failed to add radar {config.radar_id}: {e}")
            return False

    def get_frame(self, radar_id: str, timestamp: float = 0.0) -> RadarFrame:
        """Get current radar frame from sensor.

        Args:
            radar_id: ID of the radar sensor
            timestamp: Current simulation timestamp

        Returns:
            RadarFrame with sensor data
        """
        config = self.configs.get(radar_id)
        if config is None:
            logger.warning(f"Unknown radar: {radar_id}")
            return RadarFrame(timestamp=timestamp, radar_id=radar_id)

        radar = self.radars.get(radar_id)
        if radar is None:
            # Return synthetic data for standalone testing
            return self._generate_synthetic_frame(radar_id, timestamp)

        try:
            # Get data from RTX Radar sensor
            point_cloud = radar.get_point_cloud()
            range_doppler = radar.get_range_doppler_map()
            range_azimuth = radar.get_range_azimuth_map()
            targets = radar.get_targets()

            return RadarFrame(
                timestamp=timestamp,
                radar_id=radar_id,
                point_cloud=point_cloud if point_cloud is not None else np.zeros((0, 4)),
                range_doppler_map=range_doppler if range_doppler is not None else np.zeros((256, 64)),
                range_azimuth_map=range_azimuth if range_azimuth is not None else np.zeros((256, 64)),
                targets=targets if targets is not None else np.zeros((0, 5)),
                num_detections=len(targets) if targets is not None else 0,
            )
        except Exception as e:
            logger.error(f"Failed to get frame from {radar_id}: {e}")
            return self._generate_synthetic_frame(radar_id, timestamp)

    def _generate_synthetic_frame(
        self,
        radar_id: str,
        timestamp: float,
        scenario: str = "normal",
    ) -> RadarFrame:
        """Generate synthetic radar frame for testing.

        Args:
            radar_id: Radar ID
            timestamp: Timestamp
            scenario: One of "normal", "fall", "rehab_bad_posture", "chest_abnormal"

        Returns:
            Synthetic RadarFrame
        """
        config = self.configs.get(radar_id, RadarConfig())

        # Generate range-doppler map based on scenario
        rd_map = np.random.randn(config.num_range_bins, config.num_doppler_bins) * 0.1
        ra_map = np.random.randn(config.num_range_bins, config.num_azimuth_bins) * 0.1

        # Add scenario-specific signatures
        if scenario == "fall":
            # High velocity signature at medium range (falling motion)
            rd_map[50:80, 40:60] += np.random.randn(30, 20) * 2.0 + 3.0
            ra_map[50:80, 25:40] += np.random.randn(30, 15) * 1.5
            num_detections = np.random.randint(15, 30)
        elif scenario == "rehab_bad_posture":
            # Rhythmic pattern with irregular amplitude
            for i in range(5):
                offset = int(timestamp * 10 + i * 10) % 50
                rd_map[20+offset:30+offset, 30:35] += 1.5
            num_detections = np.random.randint(8, 15)
        elif scenario == "chest_abnormal":
            # Irregular micro-motion pattern
            freq = 0.5 + np.random.randn() * 0.3
            phase = timestamp * 2 * np.pi * freq
            rd_map[10:20, 32:34] += np.sin(phase) * 0.5 + np.random.randn(10, 2) * 0.3
            num_detections = np.random.randint(3, 8)
        else:  # normal
            # Stable standing/walking pattern
            rd_map[30:50, 28:36] += np.random.randn(20, 8) * 0.3 + 0.5
            num_detections = np.random.randint(5, 15)

        # Generate synthetic point cloud
        point_cloud = np.random.randn(num_detections, 6).astype(np.float32)
        point_cloud[:, 0] *= config.range_max / 3  # range
        point_cloud[:, 1] *= config.velocity_max / 2  # velocity
        point_cloud[:, 2] = np.random.uniform(-20, 0, num_detections)  # rcs
        point_cloud[:, 3] = np.random.uniform(10, 40, num_detections)  # snr
        point_cloud[:, 4] = np.random.uniform(-0.5, 0.5, num_detections)  # azimuth
        point_cloud[:, 5] = np.random.uniform(-0.3, 0.3, num_detections)  # elevation

        return RadarFrame(
            timestamp=timestamp,
            radar_id=radar_id,
            point_cloud=point_cloud,
            range_doppler_map=rd_map.astype(np.float32),
            range_azimuth_map=ra_map.astype(np.float32),
            num_detections=num_detections,
        )

    def remove_radar(self, radar_id: str) -> bool:
        """Remove a radar sensor from the scene."""
        if radar_id in self.radars:
            del self.radars[radar_id]
        if radar_id in self.configs:
            del self.configs[radar_id]
        return True

    def get_all_frames(self, timestamp: float = 0.0) -> Dict[str, RadarFrame]:
        """Get frames from all radars.

        Args:
            timestamp: Current simulation timestamp

        Returns:
            Dictionary mapping radar_id to RadarFrame
        """
        return {
            radar_id: self.get_frame(radar_id, timestamp)
            for radar_id in self.configs.keys()
        }


# =============================================================================
# USD Prim Creation Functions
# =============================================================================


def create_radar_prim(
    parent: str = "/World",
    path: str = "/World/mmwave_radar",
    config_path: Optional[str] = None,
) -> None:
    """Create an RTX Radar sensor prim in the scene.

    Uses Isaac Sim / Sensor RTX commands to create a radar sensor.
    """
    if not OMNIVERSE_AVAILABLE:
        logger.warning("Omniverse not available, cannot create radar prim")
        return

    carb.log_info(f"[mmwave.fall.radar] Creating RTX Radar at {path} under {parent}")

    kwargs = {
        "path": path,
        "parent": parent,
    }
    if config_path is not None:
        kwargs["config"] = config_path

    try:
        omni.kit.commands.execute("IsaacSensorCreateRtxRadar", **kwargs)
    except Exception as exc:
        carb.log_error(f"[mmwave.fall.radar] Failed to create RTX Radar: {exc}")


def attach_radar_to_prim(radar_path: str, target_prim_path: str) -> None:
    """Attach / parent the radar sensor under a target prim."""
    if not OMNIVERSE_AVAILABLE:
        return

    stage = omni.usd.get_context().get_stage()
    radar_prim = stage.GetPrimAtPath(radar_path)
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not radar_prim or not target_prim:
        carb.log_warn(
            f"[mmwave.fall.radar] Cannot attach radar {radar_path} to {target_prim_path}"
        )
        return

    try:
        omni.kit.commands.execute(
            "MovePrimCommand",
            path_from=radar_path,
            path_to=f"{target_prim_path}/mmwave_radar".rstrip("/"),
        )
        carb.log_info(f"[mmwave.fall.radar] Attached radar under {target_prim_path}")
    except Exception as exc:
        carb.log_error(f"[mmwave.fall.radar] Failed to attach radar: {exc}")


def _create_coverage_cone(
    stage: Any,
    parent_path: str,
    name: str,
    radius: float,
    angle_deg: float,
) -> None:
    """Create a semi-transparent cone to visualize radar coverage."""
    if not OMNIVERSE_AVAILABLE:
        return

    prim_path = f"{parent_path}/{name}"
    cone = UsdGeom.Cone.Define(stage, prim_path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(radius)
    # Rotate to point along +X
    xform = UsdGeom.Xformable(cone)
    xform.AddRotateYOp().Set(90.0)
    # Basic color
    cone.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 1.0)])
    cone.CreateDisplayOpacityAttr().Set(0.2)


def create_multi_radar_array(
    parent: str = "/World/Radars",
    positions: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> None:
    """Spawn multiple radars with coverage visualization."""
    if not OMNIVERSE_AVAILABLE:
        logger.warning("Omniverse not available, cannot create radar array")
        return

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_warn("[mmwave.fall.radar] No USD stage available")
        return

    group_prim = stage.GetPrimAtPath(parent)
    if not group_prim:
        group_prim = UsdGeom.Xform.Define(stage, parent).GetPrim()

    if positions is None:
        positions = [
            (-4.0, 2.5, 0.0),
            (0.0, 2.5, -4.0),
            (4.0, 2.5, 0.0),
        ]

    for idx, (x, y, z) in enumerate(positions):
        radar_path = f"{parent}/Radar_{idx:02d}"
        create_radar_prim(parent=parent, path=radar_path)

        xform = UsdGeom.Xformable(stage.GetPrimAtPath(radar_path))
        xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))
        _create_coverage_cone(stage, radar_path, "coverage", radius=6.0, angle_deg=90.0)


def create_radar_prim_from_config(
    stage: Any,
    prim_path: str,
    config: RadarConfig,
) -> Optional[Any]:
    """Create radar prim in USD stage from RadarConfig."""
    if not OMNIVERSE_AVAILABLE:
        logger.warning("Cannot create radar prim: Omniverse not available")
        return None

    try:
        # Create Xform for radar
        radar_xform = UsdGeom.Xform.Define(stage, prim_path)
        xformable = UsdGeom.Xformable(radar_xform)

        # Set position
        xformable.AddTranslateOp().Set(Gf.Vec3d(*config.position))

        # Set rotation (pitch, yaw, roll)
        xformable.AddRotateXYZOp().Set(Gf.Vec3f(*config.rotation))

        # Add custom attributes
        prim = radar_xform.GetPrim()
        prim.CreateAttribute("radar:id", Sdf.ValueTypeNames.String).Set(config.radar_id)
        prim.CreateAttribute("radar:name", Sdf.ValueTypeNames.String).Set(config.name)
        prim.CreateAttribute("radar:fov_h", Sdf.ValueTypeNames.Float).Set(config.fov_horizontal)
        prim.CreateAttribute("radar:fov_v", Sdf.ValueTypeNames.Float).Set(config.fov_vertical)
        prim.CreateAttribute("radar:range_max", Sdf.ValueTypeNames.Float).Set(config.range_max)

        logger.info(f"Created radar prim: {prim_path}")
        return radar_xform

    except Exception as e:
        logger.error(f"Failed to create radar prim: {e}")
        return None


# =============================================================================
# YAML Configuration Loading
# =============================================================================


def load_radars_from_yaml(yaml_path: Path) -> List[RadarConfig]:
    """Load radar configurations from YAML file.

    Args:
        yaml_path: Path to facility YAML configuration

    Returns:
        List of RadarConfig objects
    """
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    radars = []
    for radar_data in config.get("radars", []):
        pos = radar_data.get("position", {})
        rot = radar_data.get("rotation", {})
        fov = radar_data.get("fov", {})

        radars.append(RadarConfig(
            radar_id=radar_data.get("id", f"radar_{len(radars)}"),
            name=radar_data.get("name", "Unnamed Radar"),
            position=(pos.get("x", 0), pos.get("y", 2.5), pos.get("z", 0)),
            rotation=(rot.get("pitch", 0), rot.get("yaw", 0), rot.get("roll", 0)),
            fov_horizontal=fov.get("horizontal", 120.0),
            fov_vertical=fov.get("vertical", 60.0),
            range_max=radar_data.get("range_m", 10.0),
            coverage_zones=radar_data.get("coverage_zones", []),
            alert_priority=radar_data.get("alert_priority", "medium"),
        ))

    return radars


# =============================================================================
# Utility Functions
# =============================================================================


def _euler_to_quat(euler_deg: Tuple[float, float, float]) -> np.ndarray:
    """Convert Euler angles (degrees) to quaternion.

    Args:
        euler_deg: (pitch, yaw, roll) in degrees

    Returns:
        Quaternion [w, x, y, z]
    """
    pitch, yaw, roll = np.radians(euler_deg)

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])
