#!/usr/bin/env python3
"""RTX Radar sensor integration for mmWave fall detection.

This module provides wrappers around Isaac Sim's RTX Radar sensor APIs
to simulate mmWave radar sensing in the 赤土崎多功能館 scene.

Reference: https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_rtx_based_radar.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Omniverse/Isaac Sim modules
try:
    import omni.kit.app
    from omni.isaac.core import World
    from omni.isaac.core.prims import XFormPrim
    from pxr import Gf, Sdf, Usd, UsdGeom

    # RTX Radar imports
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

    OMNIVERSE_AVAILABLE = True
except ImportError:
    OMNIVERSE_AVAILABLE = False
    ISAACSIM_SENSORS_AVAILABLE = False
    logger.warning("Omniverse/Isaac Sim not available. Running in standalone mode.")


@dataclass
class RadarConfig:
    """Configuration for RTX Radar sensor."""

    # Radar identification
    radar_id: str = "radar_001"
    name: str = "mmWave Radar"

    # Position and orientation
    position: tuple[float, float, float] = (0.0, 2.5, 0.0)  # x, y, z in meters
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # pitch, yaw, roll in degrees

    # Radar parameters (mimicking TI IWR6843ISK)
    fov_horizontal: float = 120.0  # degrees
    fov_vertical: float = 60.0     # degrees
    range_max: float = 10.0        # meters
    range_min: float = 0.1         # meters
    range_resolution: float = 0.04 # meters (~4cm)
    velocity_max: float = 10.0     # m/s
    velocity_resolution: float = 0.1  # m/s

    # Sensor output configuration
    num_range_bins: int = 256
    num_doppler_bins: int = 64
    num_azimuth_bins: int = 64

    # Update rate
    update_rate_hz: float = 20.0


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

        Returns:
            Feature vector of shape [256] or raw maps
        """
        # Option 1: Use flattened range-doppler map (subsampled)
        if flatten:
            # Subsample to 16x16 = 256 features
            rd_subsampled = self.range_doppler_map[::16, ::4]  # [16, 16]
            return rd_subsampled.flatten().astype(np.float32)

        # Option 2: Return full maps
        return np.stack([
            self.range_doppler_map,
            self.range_azimuth_map,
        ], axis=0).astype(np.float32)


class RadarSensorManager:
    """Manager for RTX Radar sensors in Isaac Sim scene."""

    def __init__(self, world: World | None = None):
        """Initialize radar sensor manager.

        Args:
            world: Isaac Sim World instance (optional, for standalone mode)
        """
        self.world = world
        self.radars: dict[str, Any] = {}
        self.configs: dict[str, RadarConfig] = {}
        self._stage: Usd.Stage | None = None

    @property
    def stage(self) -> Usd.Stage | None:
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
        if not ISAACSIM_SENSORS_AVAILABLE:
            logger.warning(f"Cannot add radar {config.radar_id}: RTX Radar not available")
            self.configs[config.radar_id] = config
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
            self.configs[config.radar_id] = config

            logger.info(f"Added radar: {config.radar_id} at {config.position}")
            return True

        except Exception as e:
            logger.error(f"Failed to add radar {config.radar_id}: {e}")
            self.configs[config.radar_id] = config
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
        elif scenario == "rehab_bad_posture":
            # Rhythmic pattern with irregular amplitude
            for i in range(5):
                offset = int(timestamp * 10 + i * 10) % 50
                rd_map[20+offset:30+offset, 30:35] += 1.5
        elif scenario == "chest_abnormal":
            # Irregular micro-motion pattern
            freq = 0.5 + np.random.randn() * 0.3  # Irregular breathing
            phase = timestamp * 2 * np.pi * freq
            rd_map[10:20, 32:34] += np.sin(phase) * 0.5 + np.random.randn(10, 2) * 0.3
        else:  # normal
            # Stable standing/walking pattern
            rd_map[30:50, 28:36] += np.random.randn(20, 8) * 0.3 + 0.5

        return RadarFrame(
            timestamp=timestamp,
            radar_id=radar_id,
            range_doppler_map=rd_map.astype(np.float32),
            range_azimuth_map=ra_map.astype(np.float32),
            num_detections=np.random.randint(1, 10),
        )

    def remove_radar(self, radar_id: str) -> bool:
        """Remove a radar sensor from the scene.

        Args:
            radar_id: ID of the radar to remove

        Returns:
            True if radar was removed
        """
        if radar_id in self.radars:
            del self.radars[radar_id]
        if radar_id in self.configs:
            del self.configs[radar_id]
        return True


def create_radar_prim(
    stage: Usd.Stage,
    prim_path: str,
    config: RadarConfig,
) -> UsdGeom.Xform | None:
    """Create radar prim in USD stage (for scene setup).

    Args:
        stage: USD stage
        prim_path: Path for the radar prim
        config: Radar configuration

    Returns:
        UsdGeom.Xform for the radar, or None if failed
    """
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


def _euler_to_quat(euler_deg: tuple[float, float, float]) -> np.ndarray:
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


def load_radars_from_yaml(yaml_path: Path) -> list[RadarConfig]:
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
        ))

    return radars
