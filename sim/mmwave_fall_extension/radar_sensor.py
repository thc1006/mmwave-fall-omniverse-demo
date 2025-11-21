from __future__ import annotations

from typing import Optional

import carb
import omni.kit.commands


def create_radar_prim(
    parent: str = "/World",
    path: str = "/World/mmwave_radar",
    config_path: Optional[str] = None,
) -> None:
    """Create an RTX Radar sensor prim in the scene.

    This uses Isaac Sim / Sensor RTX commands to create a radar sensor that writes
    its results into an RtxSensorCpu buffer. Please adapt the arguments to match
    the current Isaac Sim version and documentation.
    """
    carb.log_info(f"[mmwave.fall.radar] Creating RTX Radar at {path} under {parent}")

    kwargs = {
        "path": path,
        "parent": parent,
    }
    # Many Isaac Sim examples load configuration (FOV, resolution, etc.) from JSON.
    # You can pass a config file if your Isaac version supports it.
    if config_path is not None:
        kwargs["config"] = config_path

    try:
        omni.kit.commands.execute("IsaacSensorCreateRtxRadar", **kwargs)
    except Exception as exc:
        carb.log_error(f"[mmwave.fall.radar] Failed to create RTX Radar: {exc}")


def attach_radar_to_prim(radar_path: str, target_prim_path: str) -> None:
    """Attach / parent the radar sensor under a target prim (e.g., a wall mount or robot).

    This is useful if you want the radar to move with the avatar or a fixture.
    """
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    radar_prim = stage.GetPrimAtPath(radar_path)
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not radar_prim or not target_prim:
        carb.log_warn(
            f"[mmwave.fall.radar] Cannot attach radar {radar_path} to {target_prim_path}: missing prim(s)."
        )
        return

    radar_prim.GetParent().GetChildren()
    # Simple parenting by editing the Prim path; for a serious demo, consider using Xform ops.
    try:
        omni.kit.commands.execute(
            "MovePrimCommand",
            path_from=radar_path,
            path_to=f"{target_prim_path}/mmwave_radar".rstrip("/"),
        )
        carb.log_info(
            f"[mmwave.fall.radar] Attached radar {radar_path} under {target_prim_path}."
        )
    except Exception as exc:
        carb.log_error(f"[mmwave.fall.radar] Failed to attach radar: {exc}")
