from __future__ import annotations

import os
from typing import Optional

import carb
import omni.usd
from pxr import Usd, Gf

from . import radar_sensor


DEFAULT_STAGE_PATH = "/Isaac/Environments/Simple_Room/simple_room.usd"
DEFAULT_AVATAR_PATH = "/World/FallAvatar"  # Replace with your humanoid asset path


def _get_stage() -> Usd.Stage:
    usd_context = omni.usd.get_context()
    stage = usd_context.get_stage()
    if stage is None:
        # If no stage is open, try to open the configured one.
        stage_path = os.getenv("SIM_STAGE_PATH", DEFAULT_STAGE_PATH)
        carb.log_info(f"[mmwave.fall.scene] Opening stage: {stage_path}")
        usd_context.open_stage(stage_path)
        stage = usd_context.get_stage()
    return stage


def setup_scene():
    """Set up the hall scene, avatar, and RTX Radar sensor.

    This function is called from the extension entrypoint. It assumes Isaac Sim / Kit
    is already running. You can customize the USD stage, avatar placement, and radar
    pose to match your 赤土崎多功能館 layout.
    """
    stage = _get_stage()

    # TODO: replace this with your actual humanoid asset (e.g., from Isaac Sim assets)
    avatar_prim_path = DEFAULT_AVATAR_PATH
    if not stage.GetPrimAtPath(avatar_prim_path):
        carb.log_info(f"[mmwave.fall.scene] Creating placeholder avatar at {avatar_prim_path}")
        stage.DefinePrim(avatar_prim_path, "Xform")

    # Create / attach RTX Radar sensor.
    radar_prim_path = "/World/mmwave_radar"
    radar_sensor.create_radar_prim(parent="/World", path=radar_prim_path)

    # Optionally attach radar to avatar or mount point.
    radar_sensor.attach_radar_to_prim(radar_prim_path, avatar_prim_path)

    carb.log_info("[mmwave.fall.scene] Scene setup complete. You can now run animations / data capture.")


def animate_fall_sequence(loop_count: int = 1):
    """Placeholder for driving a 'stand → walk → fall' animation sequence.

    In a production demo, this would:

    - Use animation clips / timelines to move the avatar through the hall
    - Include at least two categories: normal motion vs fall (跌倒)
    - Optionally randomize initial positions / headings

    For now this function is left as a hook that you can call from Script Editor
    or from the `record_fall_data.py` capture script.
    """
    carb.log_info(f"[mmwave.fall.scene] animate_fall_sequence called with loop_count={loop_count}. TODO: implement animation.")
