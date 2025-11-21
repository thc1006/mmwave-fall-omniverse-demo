from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import carb
import omni.kit.commands
import omni.usd
from pxr import UsdGeom, Gf


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
    if config_path is not None:
        kwargs["config"] = config_path

    try:
        omni.kit.commands.execute("IsaacSensorCreateRtxRadar", **kwargs)
    except Exception as exc:
        carb.log_error(f"[mmwave.fall.radar] Failed to create RTX Radar: {exc}")


def attach_radar_to_prim(radar_path: str, target_prim_path: str) -> None:
    """Attach / parent the radar sensor under a target prim (e.g., a wall mount or robot)."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    radar_prim = stage.GetPrimAtPath(radar_path)
    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not radar_prim or not target_prim:
        carb.log_warn(
            f"[mmwave.fall.radar] Cannot attach radar {radar_path} to {target_prim_path}: missing prim(s)."
        )
        return

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


def _create_coverage_cone(stage, parent_path: str, name: str, radius: float, angle_deg: float) -> None:
    """Create a simple semi-transparent cone to visualize radar coverage.

    This uses a mesh cone as a visual approximation of the mmWave radar field of view.
    """
    prim_path = f"{parent_path}/{name}"
    cone = UsdGeom.Cone.Define(stage, prim_path)
    cone.CreateRadiusAttr(radius)
    cone.CreateHeightAttr(radius)
    # Rotate to point along +X (or customize as needed)
    xform = UsdGeom.Xformable(cone)
    xform.AddRotateYOp().Set(90.0)
    # Basic color; you can override with a material in the DCC
    cone.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.8, 1.0)])
    cone.CreateDisplayOpacityAttr().Set(0.2)


def create_multi_radar_array(
    parent: str = "/World/Radars",
    positions: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> None:
    """Spawn multiple radars with coverage visualization.

    Parameters
    ----------
    parent:
        Parent path under which all radars and coverage cones will be grouped.
    positions:
        Optional list of (x, y, z) positions. If omitted, a small default
        array will be created for demo purposes.
    """
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_warn("[mmwave.fall.radar] No USD stage available; cannot create multi-radar array.")
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

        # Place the radar prim at the specified position.
        xform = UsdGeom.Xformable(stage.GetPrimAtPath(radar_path))
        xform.AddTranslateOp().Set(Gf.Vec3f(x, y, z))

        # Visualize coverage with a cone under a child prim.
        _create_coverage_cone(stage, radar_path, "coverage", radius=6.0, angle_deg=90.0)
