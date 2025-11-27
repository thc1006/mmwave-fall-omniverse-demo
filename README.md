# mmWave Fall Detection Demo

基於 NVIDIA Omniverse / Isaac Sim 的毫米波雷達跌倒偵測系統展示專案。

## 概述

本專案透過 RTX Radar 模擬毫米波感測器資料，搭配深度學習模型進行跌倒偵測。目標場景為「赤土崎多功能館」長照日照中心，當偵測到跌倒事件時，系統可即時通知照護人員或家屬。

## 技術架構

| 層級 | 技術 |
|------|------|
| 模擬 | NVIDIA Isaac Sim, RTX Radar, USD 場景 |
| 模型 | PyTorch (MLP / CNN / LSTM) |
| API | FastAPI + WebSocket |
| 前端 | TypeScript / Vite |
| 部署 | Docker Compose |

## 快速開始

```bash
# 安裝相依套件
make install

# 從 YAML 產生 USD 場景
make generate-usd

# 啟動 API 伺服器 (開發模式)
make api-dev

# 訓練模型
make train
```

API 文件：http://localhost:8000/docs

## 目錄結構

```
.
├── sim/                    # Omniverse 擴充套件與場景
│   ├── mmwave_fall_extension/  # Kit Extension
│   └── usd/                    # USD 場景產生器
├── ml/                     # 機器學習模組
│   ├── fallnet_model.py        # 模型定義 (MLP/CNN/LSTM)
│   ├── train_fallnet.py        # 訓練腳本
│   └── data/                   # 訓練資料 (.npz)
├── services/api/           # FastAPI 後端
│   ├── main.py                 # API 端點
│   └── websocket_manager.py    # WebSocket 管理
├── frontend/               # 前端儀表板
├── facility/               # 場地配置 YAML
├── infra/                  # Docker 與基礎設施
└── tests/                  # 測試
```

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/health` | 健康檢查 |
| GET | `/stats` | 統計摘要 |
| POST | `/predict` | 批次推論 |
| POST | `/events/from_prediction` | 單筆推論並記錄事件 |
| GET | `/events/recent` | 取得近期事件 |
| POST | `/alerts` | 發送警報通知 |
| WS | `/ws/events` | 即時事件串流 |

## 模型類別

系統支援四種偵測類別：

| 標籤 | 說明 |
|------|------|
| `normal` | 正常活動 |
| `fall` | 跌倒事件 |
| `rehab_bad_posture` | 復健姿勢不良 |
| `chest_abnormal` | 胸腔/呼吸異常 |

## 場地配置

場地設定檔位於 `facility/chih_tu_qi_floor1_ltc.yaml`，定義：

- 空間區域 (zones)：入口大廳、失智專區、復健室、餐廳等
- 雷達配置 (radars)：位置、FOV、覆蓋區域
- 模擬情境 (scenarios)：正常營運、跌倒事件模擬

## Docker 部署

```bash
# 啟動 API 容器
docker compose up -d

# 檢視日誌
docker compose logs -f

# 停止
docker compose down
```

## Makefile 指令

```bash
make help          # 顯示所有可用指令
make api-dev       # 啟動開發伺服器
make generate-usd  # 產生 USD 場景
make train         # 訓練模型 (100 epochs)
make train-quick   # 快速訓練 (10 epochs)
make sim-fall      # 執行跌倒模擬
make test          # 執行測試
```

## 環境需求

- Python 3.10+
- NVIDIA Isaac Sim (需另行安裝)
- CUDA (建議)
- Node.js 18+ (前端開發)

## 授權

本專案僅供展示與研究用途。
