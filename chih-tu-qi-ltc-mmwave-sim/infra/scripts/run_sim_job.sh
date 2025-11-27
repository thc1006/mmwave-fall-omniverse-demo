#!/usr/bin/env bash
# =============================================================================
# run_sim_job.sh - 在 Isaac Sim Docker 容器內執行模擬任務
# =============================================================================
# 用法:
#   ./infra/scripts/run_sim_job.sh <scenario> [options]
#
# 範例:
#   ./infra/scripts/run_sim_job.sh normal_operation
#   ./infra/scripts/run_sim_job.sh fall_incident --episodes 100
#   ./infra/scripts/run_sim_job.sh wandering_alert --output ml/data/wandering
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTAINER_NAME="chih-tu-qi-ltc-mmwave-sim-isaac-headless-1"

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# 預設參數
SCENARIO="${1:-normal_operation}"
EPISODES="${EPISODES:-10}"
FRAMES="${FRAMES:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-ml/data}"
USD_STAGE="${USD_STAGE:-sim/usd/chih_tu_qi_floor1_ltc.usd}"
CONFIG_FILE="${CONFIG_FILE:-facility/chih_tu_qi_floor1_ltc.yaml}"

# 解析額外參數
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --stage)
            USD_STAGE="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 <scenario> [options]"
            echo ""
            echo "可用 scenarios:"
            echo "  normal_operation  - 正常營運模擬 (09:00-17:00)"
            echo "  fall_incident     - 跌倒事件模擬"
            echo "  wandering_alert   - 徘徊警報模擬"
            echo ""
            echo "選項:"
            echo "  --episodes N      - 每個類別的 episode 數量 (預設: 10)"
            echo "  --frames N        - 每個 episode 的幀數 (預設: 128)"
            echo "  --output DIR      - 輸出目錄 (預設: ml/data)"
            echo "  --stage USD_FILE  - USD 場景檔案路徑"
            echo "  --help, -h        - 顯示此說明"
            exit 0
            ;;
        *)
            warn "未知參數: $1"
            shift
            ;;
    esac
done

# 檢查 Docker 容器是否運行
check_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
        error "Isaac Sim 容器未運行。請先執行: docker compose -f infra/docker-compose.isaac-headless.yml up -d"
    fi
    info "容器 ${CONTAINER_NAME} 正在運行"
}

# 生成 USD 場景 (如果不存在)
generate_usd_if_needed() {
    local usd_path="${PROJECT_ROOT}/${USD_STAGE}"
    if [[ ! -f "${usd_path}" ]]; then
        info "USD 場景不存在，正在生成..."
        docker exec "${CONTAINER_NAME}" /isaac-sim/python.sh \
            sim/usd/generate_floor1_from_yaml.py \
            --config "${CONFIG_FILE}" \
            --out "${USD_STAGE}"
        info "USD 場景已生成: ${USD_STAGE}"
    else
        info "使用現有 USD 場景: ${USD_STAGE}"
    fi
}

# 執行模擬任務
run_simulation() {
    info "開始執行模擬任務..."
    info "  Scenario:  ${SCENARIO}"
    info "  Episodes:  ${EPISODES}"
    info "  Frames:    ${FRAMES}"
    info "  Output:    ${OUTPUT_DIR}"
    info "  USD Stage: ${USD_STAGE}"

    # 在容器內執行 record_fall_data.py
    docker exec "${CONTAINER_NAME}" /isaac-sim/python.sh \
        sim/mmwave_fall_extension/record_fall_data.py \
        --output-dir "${OUTPUT_DIR}" \
        --episodes "${EPISODES}" \
        --frames "${FRAMES}" \
        --scenario "${SCENARIO}" \
        --stage "${USD_STAGE}" \
        --config "${CONFIG_FILE}"

    info "模擬任務完成！"
    info "資料已儲存至: ${OUTPUT_DIR}/${SCENARIO}/"
}

# 顯示結果摘要
show_summary() {
    local output_path="${PROJECT_ROOT}/${OUTPUT_DIR}/${SCENARIO}"
    if [[ -d "${output_path}" ]]; then
        local count=$(find "${output_path}" -name "*.npz" 2>/dev/null | wc -l)
        info "已生成 ${count} 個 .npz 檔案"
    fi
}

# 主程式
main() {
    info "=== Isaac Sim 模擬任務執行器 ==="
    info "Scenario: ${SCENARIO}"
    echo ""

    check_container
    generate_usd_if_needed
    run_simulation
    show_summary

    echo ""
    info "=== 任務完成 ==="
}

main "$@"
