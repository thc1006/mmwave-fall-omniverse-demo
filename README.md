# mmWave Fall Detection Demo – Omniverse + Isaac Sim + RTX Radar

This project is a **Claude Code–friendly** scaffold for a realistic demo of:

- A 3D indoor scene (e.g., 新竹市「赤土崎多功能館」簡化版) built on **NVIDIA Omniverse / Isaac Sim**  
- A simulated **mmWave (millimeter‑wave) RTX Radar sensor** that observes people moving and falling  
- A simple **ML classifier** that detects falls from radar data and triggers notifications  
- A set of **Claude Code Skills + slash commands** to help you develop and iterate entirely from the CLI

The goal is *not* to be production‑ready, but to give judges a “wow” demo that clearly shows:

1. A 3D scene where a human avatar walks, then falls (跌倒偵測, fall detection)  
2. An RTX Radar sensor that “sees” the motion and produces synthetic radar data  
3. A model that classifies normal vs fall sequences  
4. A small API that turns model outputs into structured events (for pushing to 家屬 / 照護者, etc.)  
5. Tight integration with **Claude Code CLI** so you can drive the whole loop with a few commands

---

## Repository structure

```text
mmwave-fall-omniverse-demo/
├─ CLAUDE.md                  # Main Claude Code configuration and project rules
├─ README.md                  # This file
├─ requirements.txt           # Python deps (ML + API; Isaac Sim installed separately)
├─ .env.example               # Example env vars for paths and ports
├─ sim/                       # NVIDIA Omniverse / Isaac Sim side
│  ├─ extension.toml          # Omniverse Kit extension config
│  ├─ mmwave_fall_extension/
│  │  ├─ __init__.py          # omni.ext.IExt entrypoint
│  │  ├─ scene.py             # Set up hall, avatar, animation hooks
│  │  ├─ radar_sensor.py      # RTX Radar creation / attachment helpers
│  │  └─ record_fall_data.py  # Isaac Sim script to record radar data for ML
│  └─ config/
│     └─ radar_highres_stub.json  # Stub RTX Radar config (replace with real JSON)
├─ ml/
│  ├─ fallnet_model.py        # Simple PyTorch model for fall vs non‑fall
│  ├─ train_fallnet.py        # Training script using synthetic / recorded radar data
│  └─ data/
│     └─ .gitkeep
├─ services/
│  └─ api/
│     └─ main.py              # FastAPI service: /predict, /health, etc.
└─ claude/
   ├─ skills/
   │  ├─ mmwave-fall-sim/
   │  │  └─ SKILL.md          # Skill for Omniverse / RTX Radar simulation work
   │  └─ mmwave-fall-ml/
   │     └─ SKILL.md          # Skill for ML / model training / API wiring
   └─ commands/
      ├─ setup-env.md         # /setup-env
      ├─ run-sim.md           # /run-sim
      ├─ train-model.md       # /train-model
      └─ run-demo.md          # /run-demo
```

---

## High‑level workflow

1. **Isaac Sim / Omniverse side (`sim/`)**
   - A Kit extension (`sim/extension.toml`, `mmwave_fall_extension/__init__.py`) that, when enabled, can:
     - Load a simple indoor hall USD stage (你可以替換成赤土崎多功能館的簡化模型)
     - Spawn or reference a humanoid avatar
     - Attach one or more **RTX Radar** sensors using the `omni.sensors.nv.radar` / `isaacsim.sensors.rtx` APIs
   - A script `record_fall_data.py` that runs under Isaac Sim’s Python environment to:
     - Play back episodes: walking, standing, falling (跌倒), sitting, etc.
     - Capture radar outputs from the sensor (via the `RtxSensorCpu` buffer) into `.npz` / `.npy` files
     - Label episodes as `fall` or `normal` for downstream ML

2. **ML side (`ml/`)**
   - `fallnet_model.py` defines a small PyTorch network that ingests radar frames or pre‑extracted features.
   - `train_fallnet.py`:
     - Loads recorded radar data from `ml/data/`
     - Splits into train/val
     - Trains a classifier and saves an artifact `ml/fallnet.pt`

3. **Service / API side (`services/api/`)**
   - `main.py` exposes a **FastAPI** server that:
     - Loads `fallnet.pt` on startup
     - Offers `/predict` for single or batched radar sequences
     - Can be called from Omniverse (Python) or any other client to trigger notifications

4. **Claude Code integration (`CLAUDE.md`, `claude/`)**
   - `CLAUDE.md` explains:
     - Repo layout and responsibilities for Claude
     - Which directories map to **simulation**, **ML**, and **API**
     - Which commands to use for setup, training, and running the demo
   - `claude/skills/` contains **Agent Skills** (SKILL.md) that give Claude extra context and procedures for:
     - Isaac Sim / RTX Radar / Omniverse extension work
     - Radar ML training and evaluation
   - `claude/commands/` defines **custom slash commands** for Claude Code:
     - `/setup-env` – create venv and install deps
     - `/run-sim` – help you run Isaac Sim with the extension & data recording script
     - `/train-model` – run `ml/train_fallnet.py`
     - `/run-demo` – run the FastAPI backend and describe how to hook it into Omniverse

---

## How this maps to your recorded workflow

We explicitly support the first workflow you described:

> 毫米波（millimeter‑wave, mmWave）的跌倒偵測… 毫米波的訊號打到人身上，構築出這個人的外觀，根據外觀判斷是否跌倒，若跌倒就即時推播給不在場家屬或在場照護人員。

In this scaffold:

- The **RTX Radar** sensor in Isaac Sim is your mmWave radar proxy. It produces physically‑based reflections from the avatar and environment.
- The **fall animation** episodes (站立→行走→跌倒) create distinctive radar signatures that you record via `record_fall_data.py`.
- `train_fallnet.py` turns those signatures into a supervised dataset and trains a classifier.
- `services/api/main.py` turns the classifier into an online service that can inform:
  - “不在場的家屬” – via mobile push / messaging (you can integrate SMS / LINE later)
  - “在場的照護人員” – via a local dashboard or in‑scene UI in Omniverse

**Important:** This repository intentionally stops at a **clean engineering boundary**:
- It is *ready* for you (or Claude Code) to fill in Isaac Sim–specific details (asset paths, radar config).
- It provides enough structure that you can focus on: scene design, animation, data quality, and ML, instead of wiring everything from scratch.
