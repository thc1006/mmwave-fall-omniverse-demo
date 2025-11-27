# 環境復現指南

本文件記錄所有被 `.gitignore` 忽略但專案運行所需的檔案、資料集與環境設定。

---

## 1. 資料集 (Datasets)

### 1.1 FallAllD Dataset

- **位置**: `FallAllD/`
- **大小**: 約 1.1 GB
- **說明**: 跌倒與日常活動的開放資料集，包含 IMU 感測器資料
- **下載來源**:
  - 官方連結 (可能已失效): http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/
  - 論文: https://www.mdpi.com/1424-8220/17/1/198
  - 備用: 聯繫論文作者取得
- **目錄結構**:
  ```
  FallAllD/
  ├── FallAllD/                    # 主資料目錄 (含 .mat 檔案)
  ├── DATASET SCRIPTS/             # 資料處理腳本
  ├── FallAllD_to_PYTHON_Structure.py
  ├── FallAllD_to_MATLAB_Structure.m
  └── *.pdf                        # 相關論文與文件
  ```

### 1.2 SisFall Dataset

- **位置**: `SisFall/SisFall_dataset/`
- **大小**: 約 724 MB
- **說明**: 跌倒偵測研究用資料集，包含加速度計與陀螺儀資料
- **下載來源**:
  - 官方連結 (可能已失效): http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/
  - 論文: https://www.mdpi.com/1424-8220/17/1/198
- **目錄結構**:
  ```
  SisFall/
  └── SisFall_dataset/
      ├── SA01/  # Subject Adult 01
      ├── SA02/
      └── ...    # 包含多個受試者資料夾
  ```

### 1.3 訓練資料 (Generated)

- **位置**: `ml/data/`
- **大小**: 約 1.6 MB (400 個 .npz 檔案)
- **說明**: 由 `ml/generate_synthetic_data.py` 或 Isaac Sim 模擬產生
- **重新產生方式**:
  ```bash
  # 方法 1: 產生合成資料
  python ml/generate_synthetic_data.py

  # 方法 2: 從 Isaac Sim 錄製
  make sim-all
  ```
- **目錄結構**:
  ```
  ml/data/
  ├── normal/           # 正常活動 (label=0)
  ├── fall/             # 跌倒事件 (label=1)
  ├── rehab_bad_posture/ # 復健姿勢不良 (label=2)
  └── chest_abnormal/   # 胸腔異常 (label=3)
  ```

---

## 2. Python 虛擬環境

### 2.1 主專案虛擬環境

- **位置**: `.venv/`
- **大小**: 約 7 GB (含 PyTorch + CUDA)
- **重建方式**:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- **或使用 Makefile**:
  ```bash
  make install
  ```

### 2.2 子專案虛擬環境

- **位置**: `chih-tu-qi-ltc-mmwave-sim/.venv/`
- **大小**: 約 7 GB
- **重建方式**:
  ```bash
  cd chih-tu-qi-ltc-mmwave-sim
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

---

## 3. Node.js 相依套件

### 3.1 Frontend Node Modules

- **位置**: `frontend/node_modules/`
- **重建方式**:
  ```bash
  cd frontend
  npm install
  ```
- **package.json 相依**:
  - `three`: ^0.160.0 (3D 視覺化)
  - `typescript`: ^5.3.0
  - `vite`: ^5.0.0

---

## 4. 環境變數設定

### 4.1 .env 檔案

- **位置**: `.env` (從 `.env.example` 複製)
- **重建方式**:
  ```bash
  cp .env.example .env
  # 根據需要修改內容
  ```
- **必要變數**:
  ```env
  MODEL_PATH=ml/fallnet_lstm.pt
  API_HOST=0.0.0.0
  API_PORT=8000
  LOG_LEVEL=INFO
  SIM_STAGE_PATH=/Isaac/Environments/Simple_Room/simple_room.usd
  CUDA_VISIBLE_DEVICES=
  ```

---

## 5. 外部軟體需求

### 5.1 NVIDIA Isaac Sim

- **版本**: 2023.1.0 或更新
- **安裝**: https://developer.nvidia.com/isaac-sim
- **說明**: 不可透過 pip 安裝，需從 NVIDIA Omniverse Launcher 安裝
- **用途**: 執行 RTX Radar 模擬、產生訓練資料

### 5.2 NVIDIA Omniverse

- **安裝**: https://www.nvidia.com/en-us/omniverse/
- **說明**: Isaac Sim 的基礎平台

### 5.3 CUDA Toolkit

- **建議版本**: 11.8 或 12.x
- **用途**: GPU 加速訓練與推論
- **驗證**:
  ```bash
  nvidia-smi
  nvcc --version
  ```

### 5.4 Docker (選用)

- **用途**: 容器化部署
- **安裝**: https://docs.docker.com/get-docker/

---

## 6. 模型檔案

### 6.1 預訓練模型

以下模型檔案已包含在 git 中：

| 檔案 | 大小 | 說明 |
|------|------|------|
| `ml/fallnet.pt` | 306 KB | MLP 模型 |
| `ml/fallnet_cnn.pt` | 914 KB | CNN 模型 |
| `ml/fallnet_lstm.pt` | 11 MB | LSTM 模型 (預設) |

### 6.2 重新訓練

如需重新訓練模型：
```bash
# 完整訓練 (100 epochs)
make train

# 快速訓練 (10 epochs)
make train-quick

# 訓練特定模型類型
make train-lstm
make train-cnn
```

---

## 7. Claude Code 設定

### 7.1 Claude 設定檔案

- **位置**: `.claude/` 目錄
- **說明**: Claude Code 的 agents、commands、skills 設定
- **重建**: 這些是專案特定設定，如需使用 Claude Code 功能需重新配置
- **目錄結構**:
  ```
  .claude/
  ├── agents/     # Claude 子代理設定
  ├── commands/   # 自訂斜線指令
  ├── skills/     # 技能定義
  └── settings.json
  ```

---

## 8. 快速復現步驟

### 完整環境設置

```bash
# 1. Clone 專案
git clone https://github.com/thc1006/mmwave-fall-omniverse-demo.git
cd mmwave-fall-omniverse-demo

# 2. 建立 Python 虛擬環境
make install

# 3. 設定環境變數
cp .env.example .env

# 4. 安裝前端相依套件
cd frontend && npm install && cd ..

# 5. (選用) 下載資料集
# - FallAllD: 聯繫論文作者
# - SisFall: 聯繫論文作者

# 6. (選用) 產生合成訓練資料
python ml/generate_synthetic_data.py

# 7. (選用) 重新訓練模型
make train

# 8. 啟動 API 伺服器
make api-dev
```

### 驗證安裝

```bash
# 檢查 API 健康狀態
curl http://localhost:8000/health

# 執行測試
make test

# 測試預測端點
make test-predict
```

---

## 9. 常見問題

### Q: PyTorch CUDA 版本不符

```bash
# 移除現有 PyTorch
pip uninstall torch

# 安裝對應 CUDA 版本
# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Q: Isaac Sim 找不到

確保 Isaac Sim 已從 Omniverse Launcher 安裝，並設定環境變數：
```bash
export ISAAC_SIM_PATH=/path/to/isaac_sim
```

### Q: 資料集下載連結失效

聯繫原始論文作者：
- FallAllD / SisFall 論文: https://www.mdpi.com/1424-8220/17/1/198

---

## 10. 檔案大小摘要

| 項目 | 大小 | 是否在 Git |
|------|------|----------|
| FallAllD 資料集 | 1.1 GB | 否 |
| SisFall 資料集 | 724 MB | 否 |
| 主虛擬環境 (.venv) | ~7 GB | 否 |
| 子專案虛擬環境 | ~7 GB | 否 |
| 訓練資料 (ml/data) | 1.6 MB | 否 |
| Node modules | ~200 MB | 否 |
| 模型檔案 (ml/*.pt) | ~12 MB | 是 |

---

*最後更新: 2024-11-27*
