---
name: mmwave-fall-sim
description: >
  Help design, maintain, and extend the NVIDIA Omniverse / Isaac Sim based
  mmWave (millimeter-wave) fall detection simulation in this repository.
---

# Skill: mmWave fall simulation (Omniverse / Isaac Sim / RTX Radar)

## When to use this skill

Claude should use this skill when the user asks you to:

- Modify or extend code under `sim/` or `sim/mmwave_fall_extension/`
- Work with Omniverse Kit extensions (`extension.toml`, `omni.ext.IExt`, etc.)
- Integrate or configure **RTX Radar** sensors (`omni.sensors.nv.radar`, `isaacsim.sensors.rtx`)
- Adjust how the 3D hall scene (赤土崎多功能館) is laid out
- Define or refine animation workflows for “stand → walk → fall (跌倒)” episodes
- Export radar data from Isaac Sim into files for downstream ML

## Objectives

- Keep the simulation code idiomatic for Isaac Sim / Omniverse.
- Minimize friction for the user when mapping the scene to a real hall layout.
- Ensure the radar setup and data export pipeline are easy to reason about and debug.
- Provide clear TODOs where the user must plug in asset paths or RTX Radar graph nodes.

## Simulation responsibilities

When editing this project, assume the following responsibilities:

- `sim/extension.toml`
  - Ensure the extension name, version, and Python module are consistent.
  - Declare dependencies on any required Omniverse extensions (e.g., `omni.sensors.nv.radar`).

- `sim/mmwave_fall_extension/__init__.py`
  - Provide a clean `omni.ext.IExt` entrypoint.
  - Avoid heavy logic here; delegate to functions in `scene.py` and `radar_sensor.py`.

- `sim/mmwave_fall_extension/scene.py`
  - Load the configured hall USD stage from the `SIM_STAGE_PATH` environment variable when set.
  - Create or reference a humanoid avatar prim to represent the monitored person.
  - Provide helper functions to trigger normal and fall animations, even if initially placeholders.

- `sim/mmwave_fall_extension/radar_sensor.py`
  - Create RTX Radar prims using the Isaac Sim commands (e.g., `IsaacSensorCreateRtxRadar`).
  - Parent / attach radar sensors to walls, ceilings, or avatars as appropriate.
  - Leave comments and docstrings that point the user to Isaac Sim’s RTX Radar documentation.

- `sim/mmwave_fall_extension/record_fall_data.py`
  - Run as a standalone Isaac Sim Python script (`./python.sh ...`).
  - Bootstrap the scene, run animation loops, and read radar outputs from the sensor.
  - Save labeled `normal` and `fall` episodes into `ml/data/normal/` and `ml/data/fall/`.

## Guardrails

- Do not attempt to install or configure Isaac Sim itself; assume it is already installed.
- Be explicit about any required manual steps (e.g., enabling certain Omniverse extensions, setting up Action Graphs).
- Prefer to paraphrase NVIDIA docs and link or reference them, rather than copying long excerpts.
- Keep examples small and composable; avoid building a huge monolithic script when a helper function will do.

## Example tasks for this skill

- “Add a second mmWave radar covering the opposite side of the hall and log both sensors.”
- “Wire record_fall_data.py to use the official RTX Radar Action Graph and export point clouds.”
- “Modify the scene so the avatar walks along a corridor before falling, to create richer training data.”
