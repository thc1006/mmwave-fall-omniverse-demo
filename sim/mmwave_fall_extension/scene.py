#!/usr/bin/env python3
"""Scene setup and animation control for mmWave fall detection simulation.

This module handles:
- Loading the 赤土崎多功能館 USD scene
- Spawning avatars (elderly, staff)
- Controlling fall animation sequences
- Integration with RTX Radar sensors
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import Omniverse/Isaac Sim modules
try:
    import carb
    import omni.kit.commands
    import omni.usd
    import omni.timeline
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    OMNIVERSE_AVAILABLE = True
except ImportError:
    OMNIVERSE_AVAILABLE = False
    logger.warning("Omniverse/Isaac Sim not available. Running in standalone mode.")

# Import radar sensor module
from . import radar_sensor


# =============================================================================
# Configuration Constants
# =============================================================================


DEFAULT_STAGE_PATH = "sim/usd/chih_tu_qi_floor1_ltc.usd"
DEFAULT_FACILITY_CONFIG = "facility/chih_tu_qi_floor1_ltc.yaml"
DEFAULT_AVATAR_PATH = "/World/Avatars"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AvatarConfig:
    """Configuration for an avatar in the scene."""

    avatar_id: str
    avatar_type: str  # "elderly", "staff"
    name: str
    position: Tuple[float, float, float]
    rotation: float = 0.0  # yaw in degrees
    fall_probability: float = 0.02
    behaviors: List[str] = None

    def __post_init__(self):
        if self.behaviors is None:
            self.behaviors = ["standing", "walking", "sitting"]


@dataclass
class AnimationState:
    """Current state of an avatar's animation."""

    avatar_id: str
    scenario: str  # "normal", "fall", "rehab_bad_posture", "chest_abnormal"
    phase: str  # "idle", "walking", "falling", "on_ground", "recovering"
    start_time: float
    duration: float
    progress: float = 0.0


# =============================================================================
# Scene Manager
# =============================================================================


class SceneManager:
    """Manages the 赤土崎多功能館 scene and avatars."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize scene manager.

        Args:
            config_path: Path to facility YAML configuration
        """
        self.config_path = Path(config_path or DEFAULT_FACILITY_CONFIG)
        self.config: Dict[str, Any] = {}
        self.avatars: Dict[str, AvatarConfig] = {}
        self.animation_states: Dict[str, AnimationState] = {}
        self._stage: Optional[Any] = None
        self._timeline: Optional[Any] = None

    @property
    def stage(self) -> Optional[Any]:
        """Get current USD stage."""
        if self._stage is not None:
            return self._stage
        if OMNIVERSE_AVAILABLE:
            try:
                usd_context = omni.usd.get_context()
                self._stage = usd_context.get_stage()
            except Exception:
                pass
        return self._stage

    @property
    def timeline(self) -> Optional[Any]:
        """Get timeline interface."""
        if self._timeline is not None:
            return self._timeline
        if OMNIVERSE_AVAILABLE:
            try:
                self._timeline = omni.timeline.get_timeline_interface()
            except Exception:
                pass
        return self._timeline

    def load_config(self) -> bool:
        """Load facility configuration from YAML.

        Returns:
            True if configuration was loaded successfully
        """
        if not self.config_path.exists():
            logger.warning(f"Configuration file not found: {self.config_path}")
            return False

        try:
            import yaml
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration: {self.config.get('name', 'Unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def setup_scene(self, stage_path: Optional[str] = None) -> bool:
        """Set up the scene from USD file or generate from configuration.

        Args:
            stage_path: Path to USD stage file

        Returns:
            True if scene was set up successfully
        """
        if not OMNIVERSE_AVAILABLE:
            logger.warning("Omniverse not available, using synthetic mode")
            return self.load_config()

        # Try to open existing stage
        stage_path = stage_path or DEFAULT_STAGE_PATH
        if Path(stage_path).exists():
            return self._open_stage(stage_path)

        # Generate stage from configuration
        logger.info(f"Stage not found at {stage_path}, generating from configuration")
        return self._generate_stage(stage_path)

    def _open_stage(self, stage_path: str) -> bool:
        """Open existing USD stage."""
        try:
            usd_context = omni.usd.get_context()
            usd_context.open_stage(stage_path)
            self._stage = usd_context.get_stage()
            logger.info(f"Opened stage: {stage_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to open stage: {e}")
            return False

    def _generate_stage(self, output_path: str) -> bool:
        """Generate USD stage from YAML configuration."""
        if not self.load_config():
            return False

        # Import USD generation module
        from ..usd import generate_floor1_from_yaml

        try:
            generate_floor1_from_yaml.generate_usd(
                self.config_path,
                Path(output_path),
            )
            return self._open_stage(output_path)
        except Exception as e:
            logger.error(f"Failed to generate stage: {e}")
            return False

    def spawn_avatars(self) -> List[str]:
        """Spawn avatars based on configuration.

        Returns:
            List of spawned avatar IDs
        """
        spawned = []

        if not self.config:
            self.load_config()

        avatar_configs = self.config.get("avatars", [])
        zones = {z["id"]: z for z in self.config.get("zones", [])}

        for avatar_group in avatar_configs:
            avatar_type = avatar_group.get("type", "elderly")
            base_name = avatar_group.get("name", "avatar")
            count = avatar_group.get("count", 1)
            spawn_zones = avatar_group.get("spawn_zones", [])
            fall_probability = avatar_group.get("fall_probability", 0.02)
            behaviors = avatar_group.get("behaviors", ["standing"])

            for i in range(count):
                avatar_id = f"{base_name}_{i:03d}"

                # Choose random spawn zone
                if spawn_zones:
                    zone_id = random.choice(spawn_zones)
                    zone = zones.get(zone_id, {})
                    rect = zone.get("rect", {"x": 0, "z": 0, "w": 5, "d": 5})

                    # Random position within zone
                    x = rect["x"] + random.random() * rect["w"]
                    z = rect["z"] + random.random() * rect["d"]
                    y = 0.0
                else:
                    x, y, z = random.uniform(0, 20), 0, random.uniform(0, 20)

                config = AvatarConfig(
                    avatar_id=avatar_id,
                    avatar_type=avatar_type,
                    name=f"{avatar_type.title()} {i+1}",
                    position=(x, y, z),
                    rotation=random.uniform(0, 360),
                    fall_probability=fall_probability,
                    behaviors=behaviors,
                )

                self.avatars[avatar_id] = config
                self._create_avatar_prim(config)
                spawned.append(avatar_id)

        logger.info(f"Spawned {len(spawned)} avatars")
        return spawned

    def _create_avatar_prim(self, config: AvatarConfig) -> Optional[Any]:
        """Create avatar prim in USD stage.

        For now, creates a simple capsule representation.
        In production, would load skeletal mesh with animations.
        """
        if not OMNIVERSE_AVAILABLE or self.stage is None:
            return None

        avatar_path = f"{DEFAULT_AVATAR_PATH}/{config.avatar_id}"

        try:
            # Create avatar xform
            avatar_xform = UsdGeom.Xform.Define(self.stage, avatar_path)
            xformable = UsdGeom.Xformable(avatar_xform)

            # Set position
            xformable.AddTranslateOp().Set(Gf.Vec3d(*config.position))
            xformable.AddRotateYOp().Set(config.rotation)

            # Create capsule for body
            body_path = f"{avatar_path}/Body"
            capsule = UsdGeom.Capsule.Define(self.stage, body_path)
            capsule.CreateRadiusAttr(0.25)
            capsule.CreateHeightAttr(1.2)

            body_xform = UsdGeom.Xformable(capsule)
            body_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0.85, 0))

            # Create sphere for head
            head_path = f"{avatar_path}/Head"
            head = UsdGeom.Sphere.Define(self.stage, head_path)
            head.CreateRadiusAttr(0.15)

            head_xform = UsdGeom.Xformable(head)
            head_xform.AddTranslateOp().Set(Gf.Vec3d(0, 1.6, 0))

            # Set color based on type
            if config.avatar_type == "elderly":
                color = Gf.Vec3f(0.8, 0.6, 0.4)  # Brown
            elif config.avatar_type == "staff":
                color = Gf.Vec3f(0.2, 0.6, 0.8)  # Blue
            else:
                color = Gf.Vec3f(0.6, 0.6, 0.6)  # Gray

            capsule.CreateDisplayColorAttr().Set([color])
            head.CreateDisplayColorAttr().Set([color])

            # Set metadata
            prim = avatar_xform.GetPrim()
            prim.CreateAttribute("avatar:id", Sdf.ValueTypeNames.String).Set(config.avatar_id)
            prim.CreateAttribute("avatar:type", Sdf.ValueTypeNames.String).Set(config.avatar_type)
            prim.CreateAttribute("avatar:fall_probability", Sdf.ValueTypeNames.Float).Set(
                config.fall_probability
            )

            logger.debug(f"Created avatar: {config.avatar_id} at {config.position}")
            return avatar_xform

        except Exception as e:
            logger.error(f"Failed to create avatar {config.avatar_id}: {e}")
            return None

    def animate_avatar(
        self,
        avatar_id: str,
        scenario: str = "normal",
        duration: float = 5.0,
    ) -> AnimationState:
        """Start animation for an avatar.

        Args:
            avatar_id: ID of the avatar to animate
            scenario: Animation scenario ("normal", "fall", "rehab_bad_posture", "chest_abnormal")
            duration: Duration of the animation in seconds

        Returns:
            AnimationState for the animation
        """
        if avatar_id not in self.avatars:
            logger.warning(f"Unknown avatar: {avatar_id}")
            return AnimationState(
                avatar_id=avatar_id,
                scenario=scenario,
                phase="idle",
                start_time=0.0,
                duration=duration,
            )

        config = self.avatars[avatar_id]
        current_time = self.timeline.get_current_time() if self.timeline else 0.0

        state = AnimationState(
            avatar_id=avatar_id,
            scenario=scenario,
            phase="starting",
            start_time=current_time,
            duration=duration,
        )

        self.animation_states[avatar_id] = state
        self._apply_animation(state)

        logger.info(f"Started '{scenario}' animation for {avatar_id} (duration: {duration}s)")
        return state

    def _apply_animation(self, state: AnimationState) -> None:
        """Apply animation to avatar based on current state."""
        if not OMNIVERSE_AVAILABLE or self.stage is None:
            return

        avatar_path = f"{DEFAULT_AVATAR_PATH}/{state.avatar_id}"
        prim = self.stage.GetPrimAtPath(avatar_path)
        if not prim:
            return

        xformable = UsdGeom.Xformable(prim)

        if state.scenario == "fall":
            self._apply_fall_animation(xformable, state)
        elif state.scenario == "rehab_bad_posture":
            self._apply_rehab_animation(xformable, state)
        elif state.scenario == "chest_abnormal":
            self._apply_chest_animation(xformable, state)
        else:
            self._apply_normal_animation(xformable, state)

    def _apply_fall_animation(self, xformable: Any, state: AnimationState) -> None:
        """Apply fall animation - avatar tips over and falls to ground."""
        # Fall phases:
        # 0.0 - 0.2: Standing, starting to lose balance
        # 0.2 - 0.6: Falling (rotation from 0 to -90 degrees)
        # 0.6 - 1.0: On ground, slight movement

        progress = state.progress
        ops = xformable.GetOrderedXformOps()

        # Find or create rotation op
        rotate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateX:
                rotate_op = op
                break

        if rotate_op is None:
            rotate_op = xformable.AddRotateXOp()

        if progress < 0.2:
            # Standing, slight wobble
            wobble = math.sin(progress * 50) * 5
            rotate_op.Set(wobble)
            state.phase = "wobbling"
        elif progress < 0.6:
            # Falling
            fall_progress = (progress - 0.2) / 0.4
            angle = -90 * fall_progress
            rotate_op.Set(angle)
            state.phase = "falling"
        else:
            # On ground
            rotate_op.Set(-90)
            state.phase = "on_ground"

    def _apply_rehab_animation(self, xformable: Any, state: AnimationState) -> None:
        """Apply rehabilitation with bad posture animation."""
        progress = state.progress
        ops = xformable.GetOrderedXformOps()

        rotate_op = None
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_op = op
                break

        if rotate_op is None:
            rotate_op = xformable.AddRotateZOp()

        # Irregular swaying motion indicating poor form
        base_angle = math.sin(progress * 4 * math.pi) * 15
        irregular = math.sin(progress * 7 * math.pi) * 8
        rotate_op.Set(base_angle + irregular)
        state.phase = "exercising"

    def _apply_chest_animation(self, xformable: Any, state: AnimationState) -> None:
        """Apply chest abnormality animation - rapid/irregular breathing."""
        progress = state.progress

        # Use scale to simulate breathing
        scale_op = None
        ops = xformable.GetOrderedXformOps()

        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op
                break

        if scale_op is None:
            scale_op = xformable.AddScaleOp()

        # Irregular rapid breathing
        breath_rate = 3.0 + math.sin(progress * 2 * math.pi) * 1.5  # Variable rate
        breath = 1.0 + math.sin(progress * breath_rate * 2 * math.pi) * 0.05
        scale_op.Set(Gf.Vec3f(1.0, breath, 1.0))
        state.phase = "breathing_abnormal"

    def _apply_normal_animation(self, xformable: Any, state: AnimationState) -> None:
        """Apply normal activity animation."""
        progress = state.progress

        # Simple idle swaying
        rotate_op = None
        ops = xformable.GetOrderedXformOps()

        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeRotateY:
                rotate_op = op
                break

        # Gentle swaying
        if rotate_op:
            current = rotate_op.Get()
            sway = math.sin(progress * 2 * math.pi) * 2
            rotate_op.Set(current + sway if current else sway)

        state.phase = "idle"

    def update_animations(self, dt: float) -> Dict[str, AnimationState]:
        """Update all active animations.

        Args:
            dt: Time delta in seconds

        Returns:
            Dictionary of updated animation states
        """
        updated = {}

        for avatar_id, state in list(self.animation_states.items()):
            # Update progress
            state.progress = min(1.0, state.progress + dt / state.duration)

            # Apply animation
            self._apply_animation(state)

            updated[avatar_id] = state

            # Clean up completed animations
            if state.progress >= 1.0:
                state.phase = "completed"

        return updated

    def get_avatar_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Get current positions of all avatars.

        Returns:
            Dictionary mapping avatar_id to (x, y, z) position
        """
        positions = {}

        for avatar_id, config in self.avatars.items():
            if OMNIVERSE_AVAILABLE and self.stage:
                avatar_path = f"{DEFAULT_AVATAR_PATH}/{avatar_id}"
                prim = self.stage.GetPrimAtPath(avatar_path)
                if prim:
                    xformable = UsdGeom.Xformable(prim)
                    # Get world transform
                    transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = transform.ExtractTranslation()
                    positions[avatar_id] = (translation[0], translation[1], translation[2])
                    continue

            # Fallback to config position
            positions[avatar_id] = config.position

        return positions


# =============================================================================
# Module-level Functions (for backward compatibility)
# =============================================================================


_scene_manager: Optional[SceneManager] = None


def _get_stage() -> Optional[Any]:
    """Get current USD stage (backward compatibility)."""
    global _scene_manager
    if _scene_manager is None:
        _scene_manager = SceneManager()
    return _scene_manager.stage


def setup_scene(config_path: Optional[str] = None) -> bool:
    """Set up the hall scene, avatars, and RTX Radar sensor(s).

    This function is called from the extension entrypoint.
    """
    global _scene_manager

    _scene_manager = SceneManager(config_path)

    # Load configuration
    if not _scene_manager.load_config():
        logger.warning("Using default configuration")

    # Setup scene
    if not _scene_manager.setup_scene():
        logger.warning("Scene setup incomplete, using minimal configuration")

    # Spawn avatars
    _scene_manager.spawn_avatars()

    # Setup radar sensors from configuration
    radar_configs = radar_sensor.load_radars_from_yaml(
        Path(_scene_manager.config_path)
    ) if _scene_manager.config_path.exists() else []

    manager = radar_sensor.RadarSensorManager()
    for config in radar_configs:
        manager.add_radar(config)

    if OMNIVERSE_AVAILABLE:
        carb.log_info("[mmwave.fall.scene] Scene setup complete")

    return True


def animate_fall_sequence(
    loop_count: int = 1,
    scenario: str = "fall",
    avatar_id: Optional[str] = None,
    duration: float = 5.0,
) -> List[AnimationState]:
    """Drive a labeled animation sequence.

    Parameters
    ----------
    loop_count:
        How many times to repeat the animation.
    scenario:
        One of:
        - "fall":           normal walking followed by a fall
        - "rehab_bad_posture": rehabilitation exercise with incorrect posture
        - "chest_abnormal":  subtle thoracic / chest abnormality
        - "normal":          normal walking / standing / daily activities
    avatar_id:
        Specific avatar to animate. If None, animates a random avatar.
    duration:
        Duration of each animation loop in seconds.

    Returns
    -------
    List of AnimationState objects for the animations
    """
    global _scene_manager

    if _scene_manager is None:
        setup_scene()

    states = []

    for _ in range(loop_count):
        # Select avatar
        if avatar_id and avatar_id in _scene_manager.avatars:
            selected_id = avatar_id
        elif _scene_manager.avatars:
            selected_id = random.choice(list(_scene_manager.avatars.keys()))
        else:
            logger.warning("No avatars available for animation")
            break

        # Start animation
        state = _scene_manager.animate_avatar(
            selected_id,
            scenario=scenario,
            duration=duration,
        )
        states.append(state)

    if OMNIVERSE_AVAILABLE:
        carb.log_info(
            f"[mmwave.fall.scene] animate_fall_sequence: {loop_count} loops, scenario={scenario}"
        )

    return states


def get_scene_manager() -> SceneManager:
    """Get the global scene manager instance."""
    global _scene_manager
    if _scene_manager is None:
        _scene_manager = SceneManager()
    return _scene_manager
