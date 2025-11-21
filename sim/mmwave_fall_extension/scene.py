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
        stage_path = os.getenv("SIM_STAGE_PATH", DEFAULT_STAGE_PATH)
        carb.log_info(f"[mmwave.fall.scene] Opening stage: {stage_path}")
        usd_context.open_stage(stage_path)
        stage = usd_context.get_stage()
    return stage


def setup_scene():
    """Set up the hall scene, avatar, and RTX Radar sensor(s).

    This function is called from the extension entrypoint. It assumes Isaac Sim / Kit
    is already running. You can customize the USD stage, avatar placement, and radar
    poses to match your 赤土崎多功能館 layout.
    """
    stage = _get_stage()

    avatar_prim_path = DEFAULT_AVATAR_PATH
    if not stage.GetPrimAtPath(avatar_prim_path):
        carb.log_info(f"[mmwave.fall.scene] Creating placeholder avatar at {avatar_prim_path}")
        stage.DefinePrim(avatar_prim_path, "Xform")

    # Create one or more RTX Radar sensors. For a more impressive demo, you can
    # call `create_multi_radar_array` instead of `create_radar_prim`.
    radar_prim_path = "/World/mmwave_radar_main"
    radar_sensor.create_radar_prim(parent="/World", path=radar_prim_path)

    # Example: create additional radars covering different parts of the hall.
    radar_sensor.create_multi_radar_array(parent="/World/Radars")

    carb.log_info("[mmwave.fall.scene] Scene setup complete. You can now run animations / data capture.")


def animate_fall_sequence(loop_count: int = 1, scenario: str = "fall"):
    """Drive a labeled animation sequence.

    Parameters
    ----------
    loop_count:
        How many times to repeat the animation.
    scenario:
        One of:
        - "fall":           normal walking followed by a fall (跌倒)
        - "rehab_bad_posture": rehabilitation exercise with incorrect posture
        - "chest_abnormal":  subtle thoracic / chest abnormality (e.g., asymmetric motion)
        - "normal":          normal walking / standing / daily activities

    In a production demo, this function would:

    - Use animation clips / timelines to move the avatar through the hall
    - Choose a motion pattern based on `scenario`
    - Optionally randomize initial positions / headings

    For now this function is left as a hook that you can call from Script Editor
    or from the `record_fall_data.py` capture script. You may implement the actual
    animation according to your asset library.
    """
    carb.log_info(
        f"[mmwave.fall.scene] animate_fall_sequence called with loop_count={loop_count}, scenario={scenario}. "
        "TODO: implement animation clips for each scenario."
    )
