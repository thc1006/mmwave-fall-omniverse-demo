#!/usr/bin/env python3
"""Generate USD scene from YAML floor plan configuration.

This script reads the facility YAML configuration and generates a USD scene
with:
- Cube geometry for each zone (walls, floors)
- Xform anchors for radar positions
- Materials for different zone types

Usage (inside Isaac Sim):
    ./python.sh sim/usd/generate_floor1_from_yaml.py \
        --config facility/chih_tu_qi_floor1_ltc.yaml \
        --out sim/usd/chih_tu_qi_floor1_ltc.usd

Usage (standalone):
    python sim/usd/generate_floor1_from_yaml.py \
        --config facility/chih_tu_qi_floor1_ltc.yaml \
        --out sim/usd/chih_tu_qi_floor1_ltc.usd
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Try to import USD libraries
try:
    from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
    USD_AVAILABLE = True
except ImportError:
    print("[WARN] pxr (USD) not available. Install via Isaac Sim or pip install usd-core")
    USD_AVAILABLE = False


# Zone type to color mapping (RGB, 0-1 range)
ZONE_COLORS: dict[str, tuple[float, float, float]] = {
    "lobby": (0.9, 0.9, 0.85),           # 淺米色
    "activity_room": (0.85, 0.95, 0.85),  # 淺綠色
    "therapy_room": (0.95, 0.9, 0.95),    # 淺紫色
    "corridor": (0.95, 0.95, 0.9),        # 淺黃色
    "dining": (1.0, 0.92, 0.8),           # 淺橙色
    "kitchen": (0.9, 0.9, 0.9),           # 淺灰色
    "outdoor": (0.8, 1.0, 0.8),           # 草綠色
    "medical": (0.95, 0.85, 0.85),        # 淺粉紅
    "restroom": (0.85, 0.9, 0.95),        # 淺藍色
    "storage": (0.88, 0.88, 0.88),        # 灰色
    "office": (0.92, 0.88, 0.82),         # 米色
    "rest_area": (0.9, 0.95, 1.0),        # 淺天藍
    "default": (0.85, 0.85, 0.85),        # 預設灰色
}

# Fall risk to material opacity
FALL_RISK_OPACITY: dict[str, float] = {
    "low": 0.3,
    "medium": 0.5,
    "high": 0.7,
    "critical": 0.9,
}


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_material(
    stage: Usd.Stage,
    name: str,
    color: tuple[float, float, float],
    opacity: float = 1.0,
) -> UsdShade.Material:
    """Create a simple preview material with color."""
    mat_path = f"/World/Materials/{name}"
    material = UsdShade.Material.Define(stage, mat_path)

    # Create preview surface shader
    shader_path = f"{mat_path}/PreviewSurface"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")

    # Set diffuse color
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)

    # Connect shader to material
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    return material


def create_zone_cube(
    stage: Usd.Stage,
    zone: dict[str, Any],
    parent_path: str,
    ceiling_height: float,
) -> UsdGeom.Xform:
    """Create a cube representing a zone.

    Args:
        stage: USD stage
        zone: Zone configuration dict with id, type, rect, etc.
        parent_path: Parent prim path
        ceiling_height: Height of the zone (meters)

    Returns:
        UsdGeom.Xform for the zone
    """
    zone_id = zone["id"]
    zone_type = zone.get("type", "default")
    rect = zone["rect"]
    fall_risk = zone.get("fall_risk", "low")
    name = zone.get("name", zone_id)

    # Extract dimensions
    x = rect["x"]
    z = rect["z"]
    w = rect["w"]
    d = rect["d"]
    h = 2.5  # Fixed wall height

    # Create zone xform
    zone_path = f"{parent_path}/{zone_id}"
    zone_xform = UsdGeom.Xform.Define(stage, zone_path)

    # Set custom attributes for metadata
    prim = zone_xform.GetPrim()
    prim.CreateAttribute("zone:name", Sdf.ValueTypeNames.String).Set(name)
    prim.CreateAttribute("zone:type", Sdf.ValueTypeNames.String).Set(zone_type)
    prim.CreateAttribute("zone:fall_risk", Sdf.ValueTypeNames.String).Set(fall_risk)
    prim.CreateAttribute("zone:area_sqm", Sdf.ValueTypeNames.Float).Set(
        float(zone.get("area_sqm", w * d))
    )

    # Create floor
    floor_path = f"{zone_path}/Floor"
    floor = UsdGeom.Cube.Define(stage, floor_path)
    floor.CreateSizeAttr(1.0)

    # Scale to match dimensions (Cube is 1x1x1 centered at origin)
    floor_xform = UsdGeom.Xformable(floor)
    floor_xform.AddTranslateOp().Set(Gf.Vec3d(x + w / 2, 0.025, z + d / 2))
    floor_xform.AddScaleOp().Set(Gf.Vec3d(w, 0.05, d))

    # Apply material
    color = ZONE_COLORS.get(zone_type, ZONE_COLORS["default"])
    opacity = FALL_RISK_OPACITY.get(fall_risk, 0.3)
    mat_name = f"Mat_{zone_id}"
    material = create_material(stage, mat_name, color, 1.0)
    UsdShade.MaterialBindingAPI(floor).Bind(material)

    # Create fall risk indicator (semi-transparent overlay)
    if fall_risk in ("high", "critical"):
        indicator_path = f"{zone_path}/FallRiskIndicator"
        indicator = UsdGeom.Cube.Define(stage, indicator_path)
        indicator.CreateSizeAttr(1.0)

        indicator_xform = UsdGeom.Xformable(indicator)
        indicator_xform.AddTranslateOp().Set(Gf.Vec3d(x + w / 2, 0.1, z + d / 2))
        indicator_xform.AddScaleOp().Set(Gf.Vec3d(w - 0.2, 0.01, d - 0.2))

        # Red indicator for fall risk
        risk_color = (1.0, 0.3, 0.3) if fall_risk == "critical" else (1.0, 0.6, 0.3)
        risk_mat = create_material(stage, f"Mat_{zone_id}_Risk", risk_color, opacity)
        UsdShade.MaterialBindingAPI(indicator).Bind(risk_mat)

    # Create walls (4 sides)
    wall_thickness = 0.1
    walls = [
        ("WallNorth", x + w / 2, z + d, w, wall_thickness, h),
        ("WallSouth", x + w / 2, z, w, wall_thickness, h),
        ("WallEast", x + w, z + d / 2, wall_thickness, d, h),
        ("WallWest", x, z + d / 2, wall_thickness, d, h),
    ]

    for wall_name, wx, wz, ww, wd, wh in walls:
        wall_path = f"{zone_path}/{wall_name}"
        wall = UsdGeom.Cube.Define(stage, wall_path)
        wall.CreateSizeAttr(1.0)

        wall_xform = UsdGeom.Xformable(wall)
        wall_xform.AddTranslateOp().Set(Gf.Vec3d(wx, wh / 2, wz))
        wall_xform.AddScaleOp().Set(Gf.Vec3d(ww, wh, wd))

        # Wall material (slightly darker than floor)
        wall_color = tuple(c * 0.9 for c in color)
        wall_mat = create_material(stage, f"Mat_{zone_id}_{wall_name}", wall_color, 0.8)
        UsdShade.MaterialBindingAPI(wall).Bind(wall_mat)

    return zone_xform


def create_radar_anchor(
    stage: Usd.Stage,
    radar: dict[str, Any],
    parent_path: str,
) -> UsdGeom.Xform:
    """Create an Xform anchor for radar sensor.

    Args:
        stage: USD stage
        radar: Radar configuration dict
        parent_path: Parent prim path

    Returns:
        UsdGeom.Xform for the radar
    """
    radar_id = radar["id"]
    name = radar.get("name", radar_id)
    pos = radar["position"]
    rot = radar.get("rotation", {"pitch": 0, "yaw": 0, "roll": 0})
    fov = radar.get("fov", {"horizontal": 120, "vertical": 60})

    # Create radar xform
    radar_path = f"{parent_path}/{radar_id}"
    radar_xform = UsdGeom.Xform.Define(stage, radar_path)

    # Set transform
    xformable = UsdGeom.Xformable(radar_xform)
    xformable.AddTranslateOp().Set(Gf.Vec3d(pos["x"], pos["y"], pos["z"]))

    # Apply rotation (pitch, yaw, roll in degrees)
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(rot["pitch"], rot["yaw"], rot["roll"]))

    # Set custom attributes
    prim = radar_xform.GetPrim()
    prim.CreateAttribute("radar:name", Sdf.ValueTypeNames.String).Set(name)
    prim.CreateAttribute("radar:model", Sdf.ValueTypeNames.String).Set(
        radar.get("model", "TI IWR6843ISK")
    )
    prim.CreateAttribute("radar:fov_h", Sdf.ValueTypeNames.Float).Set(float(fov["horizontal"]))
    prim.CreateAttribute("radar:fov_v", Sdf.ValueTypeNames.Float).Set(float(fov["vertical"]))
    prim.CreateAttribute("radar:range_m", Sdf.ValueTypeNames.Float).Set(
        float(radar.get("range_m", 10.0))
    )
    prim.CreateAttribute("radar:alert_priority", Sdf.ValueTypeNames.String).Set(
        radar.get("alert_priority", "medium")
    )

    # Create visual representation (small box for the sensor)
    sensor_box_path = f"{radar_path}/SensorBody"
    sensor_box = UsdGeom.Cube.Define(stage, sensor_box_path)
    sensor_box.CreateSizeAttr(1.0)

    sensor_xform = UsdGeom.Xformable(sensor_box)
    sensor_xform.AddScaleOp().Set(Gf.Vec3d(0.1, 0.05, 0.15))  # Small box

    # Sensor material (dark gray)
    sensor_mat = create_material(stage, f"Mat_{radar_id}", (0.2, 0.2, 0.25), 1.0)
    UsdShade.MaterialBindingAPI(sensor_box).Bind(sensor_mat)

    # Create FOV visualization cone (optional)
    fov_path = f"{radar_path}/FOV_Cone"
    fov_cone = UsdGeom.Cone.Define(stage, fov_path)
    fov_cone.CreateRadiusAttr(float(radar.get("range_m", 10.0)) * 0.5)
    fov_cone.CreateHeightAttr(float(radar.get("range_m", 10.0)))

    fov_xform = UsdGeom.Xformable(fov_cone)
    fov_xform.AddRotateXOp().Set(-90)  # Point forward
    fov_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, float(radar.get("range_m", 10.0)) / 2))

    # FOV material (semi-transparent blue)
    fov_mat = create_material(stage, f"Mat_{radar_id}_FOV", (0.2, 0.5, 1.0), 0.15)
    UsdShade.MaterialBindingAPI(fov_cone).Bind(fov_mat)

    return radar_xform


def generate_usd(config_path: Path, output_path: Path) -> None:
    """Generate USD scene from YAML configuration.

    Args:
        config_path: Path to YAML configuration file
        output_path: Path to output USD file
    """
    if not USD_AVAILABLE:
        print("[ERROR] USD libraries not available. Cannot generate USD file.")
        sys.exit(1)

    # Load configuration
    config = load_config(config_path)
    print(f"[INFO] Loaded configuration: {config.get('name', 'Unknown')}")
    print(f"[INFO] Total area: {config.get('total_area_sqm', 'N/A')} m²")

    # Create new stage
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output_path))

    # Set up axis and meters per unit
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Create root xform
    world = UsdGeom.Xform.Define(stage, "/World")

    # Add metadata
    world_prim = world.GetPrim()
    world_prim.CreateAttribute("facility:name", Sdf.ValueTypeNames.String).Set(
        config.get("name", "Unknown Facility")
    )
    world_prim.CreateAttribute("facility:floor", Sdf.ValueTypeNames.Int).Set(
        config.get("floor", 1)
    )
    world_prim.CreateAttribute("facility:total_area_sqm", Sdf.ValueTypeNames.Float).Set(
        float(config.get("total_area_sqm", 0))
    )

    # Create materials group
    UsdGeom.Scope.Define(stage, "/World/Materials")

    # Create zones group
    zones_xform = UsdGeom.Xform.Define(stage, "/World/Zones")
    zones = config.get("zones", [])
    ceiling_height = config.get("ceiling_height_m", 3.5)

    print(f"[INFO] Creating {len(zones)} zones...")
    for zone in zones:
        create_zone_cube(stage, zone, "/World/Zones", ceiling_height)
        print(f"  - {zone['id']}: {zone.get('name', 'N/A')} ({zone.get('type', 'unknown')})")

    # Create radars group
    radars_xform = UsdGeom.Xform.Define(stage, "/World/Radars")
    radars = config.get("radars", [])

    print(f"[INFO] Creating {len(radars)} radar anchors...")
    for radar in radars:
        create_radar_anchor(stage, radar, "/World/Radars")
        pos = radar["position"]
        print(f"  - {radar['id']}: {radar.get('name', 'N/A')} at ({pos['x']}, {pos['y']}, {pos['z']})")

    # Save stage
    stage.GetRootLayer().Save()
    print(f"[INFO] USD scene saved to: {output_path}")
    print(f"[INFO] Stage contains:")
    print(f"  - {len(zones)} zones")
    print(f"  - {len(radars)} radar sensors")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate USD scene from YAML floor plan configuration."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output USD file path",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.out)

    if not config_path.exists():
        print(f"[ERROR] Configuration file not found: {config_path}")
        sys.exit(1)

    generate_usd(config_path, output_path)


if __name__ == "__main__":
    main()
