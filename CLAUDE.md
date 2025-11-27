# CLAUDE.md – mmWave Fall Detection (Omniverse / Isaac Sim / RTX Radar)

This file guides Claude Code when working inside the **mmwave-fall-omniverse-demo** repository.


你必須使用 .claude/agents 目錄底下的 claude subagents 來進行超高速開發，任何需要下載的檔案請幫我直接進行下載（本機密碼：Kinoshita15@）

> 你有 playwright MCP 可以直接進行使用。

- 「請打開 chih-tu-qi-ltc-mmwave-sim/architectural-floor-plans-2025.md 的『空間詳細規格-1』段落，幫我把每個空間轉成 facility/chih_tu_qi_floor1_ltc.yaml 裡的 zones 陣列，每個 zone 要有 id、type、rect（x,z,w,d，以公尺）和功能描述；順便設計 3 個 mmWave radar 的位置和 FOV，填進 radars。」
- 「用 terminal-local 幫我跑：docker compose -f infra/docker-compose.isaac-headless.yml up -d，然後確認容器有起來，再幫我寫一個指令腳本讓我可以 ./infra/scripts/run_sim_job.sh ... 叫 Isaac Sim 跑特定 scenario。」
- 「請改寫 sim/usd/generate_floor1_from_yaml.py，讀 facility/chih_tu_qi_floor1_ltc.yaml 裡的 zones / radars，用 UsdGeom.Cube 來畫出每個空間（寬 w、深 d、高固定 2.5m），再用 Xform 放 radar anchor，輸出 sim/usd/chih_tu_qi_floor1_ltc.usd。」
- 「請在 services/api/main.py 裡幫我加上 /predict，接收 List[List[float]]，呼叫 FallNet 模型做推論，並回傳 label 和每個類別的機率；順便幫我寫一個 uvicorn 啟動指令到 Makefile 裡。」


---

## 1. Project overview

This project is an advanced demo of **mmWave (millimeter‑wave) fall detection** in a realistic 3D scene using:

- **NVIDIA Omniverse / Isaac Sim** for 3D simulation and rendering  
- **RTX Radar sensors** (via `omni.sensors.nv.radar` / `isaacsim.sensors.rtx`) to generate radar‑like data  
- **PyTorch** to train a fall vs non‑fall classifier from synthetic radar sequences  
- **FastAPI** to serve model predictions as an HTTP API  
- **Claude Code Skills + Commands** to automate repetitive workflows

The target scenario is a simplified **赤土崎多功能館** hall where an mmWave radar monitors people and detects falls. When a fall is detected, downstream systems can notify family members or on‑site caregivers.

---

## 2. Technology stack

- **Python 3.10+**
- **NVIDIA Isaac Sim / Omniverse Kit** (installed separately; not via `pip`). Use the provided scripts through `./python.sh` or equivalent in the Isaac Sim install.
- **RTX Radar / Sensor RTX**:
  - Use Isaac Sim’s RTX Sensor APIs (e.g., `IsaacSensorCreateRtxRadar` / `RtxSensorCpu` buffers) to simulate mmWave sensing.
- **ML / API**:
  - `torch`, `numpy`, `pandas`, `pyarrow`
  - `fastapi`, `uvicorn[standard]`

**Important for Claude:** Do **not** try to install Isaac Sim itself with `pip`. Assume Isaac Sim and Omniverse are already installed and available on the user’s machine.

---

## 3. Source tree responsibilities

- `sim/`
  - Omniverse Kit extension (`extension.toml`, `mmwave_fall_extension/`) that:
    - Loads or references a hall USD stage
    - Spawns avatar(s) and attaches RTX Radar sensor(s)
    - Provides hooks for animating “walk → fall” sequences
    - Records radar data to `ml/data/` via `record_fall_data.py` (run inside Isaac Sim)
- `ml/`
  - `fallnet_model.py`: PyTorch model definition
  - `train_fallnet.py`: Training script that reads recorded radar sequences and outputs `fallnet.pt`
  - `data/`: Place to store `.npz` / `.npy` episodes, split into fall vs normal
- `services/api/`
  - `main.py`: FastAPI server that loads `fallnet.pt` and exposes `/predict`
- `claude/`
  - `skills/`: Agent Skills (SKILL.md) for this repo
  - `commands/`: Custom slash commands for common workflows

---

## 4. How Claude Code should behave

When operating in this repository, please:

1. **Respect the separation of concerns**
   - Keep Isaac Sim / Omniverse‑specific logic inside `sim/mmwave_fall_extension/`.
   - Keep ML code in `ml/` and HTTP APIs in `services/api/`.
2. **Prefer small, testable changes**
   - Before making large refactors, propose a short plan.
   - Implement changes in small steps and describe how to validate them (e.g., which script or endpoint to run).
3. **Use Skills and Commands where appropriate**
   - When doing significant work on the simulation, consider using the `mmwave-fall-sim` skill.
   - When doing ML or API work, consider using the `mmwave-fall-ml` skill.
   - For repetitive actions (setup, running sim, training), prefer the custom slash commands defined in `claude/commands/`.
4. **Stay close to official docs for Isaac Sim / RTX Radar**
   - When editing `record_fall_data.py` or `radar_sensor.py`, reference the official RTX Radar and Omniverse Kit extension documentation.
   - Prefer keeping API usage compatible with current Isaac Sim stable releases.

---

## 5. Setup and workflows

### 5.1 Environment setup

On a development machine with Isaac Sim installed:

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. Install Python dependencies for ML + API:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and adjust paths if needed.

Claude Code can help with this via the `/setup-env` command in `claude/commands/setup-env.md`.

### 5.2 Recording radar data in Isaac Sim

- Use Isaac Sim’s `./python.sh` (or platform equivalent) to run `sim/mmwave_fall_extension/record_fall_data.py`.
- That script should:
  - Load the hall stage
  - Ensure the RTX Radar sensor is active (via `radar_sensor.create_radar_prim`)
  - Play through animation sequences for falling and non‑falling motion
  - Save labeled radar outputs in `ml/data/`

This script is intentionally left with TODOs and placeholders so you can adapt it to your specific stage and assets.

### 5.3 Training the fall detection model

- Run:

  ```bash
  python ml/train_fallnet.py --data-dir ml/data --output ml/fallnet.pt
  ```

- The script will:
  - Load `.npz` episodes
  - Perform a simple train/validation split
  - Train `FallNet` and write `fallnet.pt`

Claude Code can help by using the `/train-model` command.

### 5.4 Running the API demo

- Start the FastAPI server:

  ```bash
  uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
  ```

- Call `/predict` with radar sequences (from Isaac Sim or a recorded file) to get `fall` vs `normal` predictions.
- You can wire Omniverse Python to call this endpoint whenever a new radar frame/sequence is available.

Claude Code can assist via the `/run-demo` command.

---

## 6. Style and quality guidelines

- Use **type hints** for all new Python functions.
- Prefer **dataclasses / Pydantic models** for structured data passed between layers.
- Keep scripts **idempotent** and **configurable** via CLI args or `.env` rather than hard‑coding paths.
- When in doubt, add small docstrings and comments that reference NVIDIA documentation pages instead of copying large blocks of prose.

---

## 7. Improvements

- Replace synthetic animations with more realistic motion capture clips.
- Add additional classes (sitting, lying down, bending) and explore multi‑class classification.
- Build a web dashboard that subscribes to predictions and visualizes fall events on a floor map.
- Integrate real mmWave radar hardware by swapping Isaac Sim data loaders with live sensor drivers.
