/**
 * mmWave Fall Detection Dashboard
 * Real-time monitoring system for 赤土崎多功能館
 */

// Configuration
const CONFIG = {
    apiUrl: 'http://localhost:8002',
    wsUrl: 'ws://localhost:8002/ws',
    reconnectInterval: 3000,
    maxLogEntries: 50,
    updateInterval: 1000
};

// Zone definitions for 赤土崎多功能館 1F
const ZONES = [
    { id: 'Z01', name: '入口大廳', row: 0, col: 0 },
    { id: 'Z02', name: '接待區', row: 0, col: 1 },
    { id: 'Z03', name: '服務台', row: 0, col: 2 },
    { id: 'Z04', name: '休息區A', row: 0, col: 3 },
    { id: 'Z05', name: '休息區B', row: 0, col: 4 },
    { id: 'Z06', name: '走廊西', row: 1, col: 0 },
    { id: 'Z07', name: '活動區A', row: 1, col: 1 },
    { id: 'Z08', name: '中央大廳', row: 1, col: 2 },
    { id: 'Z09', name: '活動區B', row: 1, col: 3 },
    { id: 'Z10', name: '走廊東', row: 1, col: 4 },
    { id: 'Z11', name: '儲藏室', row: 2, col: 0 },
    { id: 'Z12', name: '會議室A', row: 2, col: 1 },
    { id: 'Z13', name: '多功能室', row: 2, col: 2 },
    { id: 'Z14', name: '會議室B', row: 2, col: 3 },
    { id: 'Z15', name: '緊急出口', row: 2, col: 4 }
];

// Radar sensor positions
const RADARS = [
    { id: 'R1', name: '雷達 R1', location: '入口區', position: { top: '15%', left: '20%' }, zones: ['Z01', 'Z02', 'Z06', 'Z07'] },
    { id: 'R2', name: '雷達 R2', location: '中央區', position: { top: '50%', left: '50%' }, zones: ['Z07', 'Z08', 'Z09', 'Z12', 'Z13', 'Z14'] },
    { id: 'R3', name: '雷達 R3', location: '東側區', position: { top: '15%', left: '80%' }, zones: ['Z04', 'Z05', 'Z09', 'Z10', 'Z14', 'Z15'] }
];

// Application State
const state = {
    ws: null,
    connected: false,
    zones: {},
    radars: {},
    detectionCount: 0,
    startTime: Date.now(),
    lastUpdate: null,
    currentAlert: null
};

// Initialize zones state
ZONES.forEach(zone => {
    state.zones[zone.id] = {
        ...zone,
        status: 'normal',
        probability: { normal: 0.95, warning: 0.04, fall: 0.01 }
    };
});

// Initialize radars state
RADARS.forEach(radar => {
    state.radars[radar.id] = {
        ...radar,
        online: true,
        lastPing: Date.now()
    };
});

/**
 * DOM Elements
 */
const elements = {
    connectionStatus: document.getElementById('connectionStatus'),
    datetime: document.getElementById('datetime'),
    floorPlan: document.getElementById('floorPlan'),
    radarCards: document.getElementById('radarCards'),
    activityLog: document.getElementById('activityLog'),
    alertModal: document.getElementById('alertModal'),
    alertZone: document.getElementById('alertZone'),
    alertTime: document.getElementById('alertTime'),
    alertProbability: document.getElementById('alertProbability'),
    alertAckBtn: document.getElementById('alertAckBtn'),
    alertDismissBtn: document.getElementById('alertDismissBtn'),
    clearLogBtn: document.getElementById('clearLogBtn'),
    gaugeNormal: document.getElementById('gaugeNormal'),
    gaugeWarning: document.getElementById('gaugeWarning'),
    gaugeFall: document.getElementById('gaugeFall'),
    gaugeNormalValue: document.getElementById('gaugeNormalValue'),
    gaugeWarningValue: document.getElementById('gaugeWarningValue'),
    gaugeFallValue: document.getElementById('gaugeFallValue'),
    statDetections: document.getElementById('statDetections'),
    statActiveZones: document.getElementById('statActiveZones'),
    statUptime: document.getElementById('statUptime'),
    statLastUpdate: document.getElementById('statLastUpdate'),
    alertSound: document.getElementById('alertSound')
};

/**
 * WebSocket Connection
 */
function connectWebSocket() {
    updateConnectionStatus('connecting');
    addLogEntry('info', '正在連接 WebSocket...');

    try {
        state.ws = new WebSocket(CONFIG.wsUrl);

        state.ws.onopen = () => {
            state.connected = true;
            updateConnectionStatus('connected');
            addLogEntry('success', 'WebSocket 連接成功');
        };

        state.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleWebSocketMessage(data);
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };

        state.ws.onclose = () => {
            state.connected = false;
            updateConnectionStatus('disconnected');
            addLogEntry('warning', 'WebSocket 連接已斷開');

            // Attempt to reconnect
            setTimeout(connectWebSocket, CONFIG.reconnectInterval);
        };

        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            addLogEntry('error', 'WebSocket 連接錯誤');
        };
    } catch (e) {
        console.error('Failed to create WebSocket:', e);
        updateConnectionStatus('disconnected');
        setTimeout(connectWebSocket, CONFIG.reconnectInterval);
    }
}

/**
 * Handle incoming WebSocket messages
 */
function handleWebSocketMessage(data) {
    state.lastUpdate = new Date();

    switch (data.type) {
        case 'zone_update':
            updateZoneStatus(data.zone_id, data.status, data.probability);
            break;
        case 'fall_alert':
            triggerFallAlert(data);
            break;
        case 'radar_status':
            updateRadarStatus(data.radar_id, data.online);
            break;
        case 'detection_result':
            handleDetectionResult(data);
            break;
        case 'heartbeat':
            // Keep-alive message
            break;
        default:
            console.log('Unknown message type:', data.type);
    }
}

/**
 * Update zone status
 */
function updateZoneStatus(zoneId, status, probability) {
    if (!state.zones[zoneId]) return;

    const previousStatus = state.zones[zoneId].status;
    state.zones[zoneId].status = status;

    if (probability) {
        state.zones[zoneId].probability = probability;
    }

    // Update DOM
    const zoneEl = document.querySelector(`[data-zone-id="${zoneId}"]`);
    if (zoneEl) {
        zoneEl.className = `zone ${status}`;
        const statusEl = zoneEl.querySelector('.zone-status');
        if (statusEl) {
            statusEl.textContent = getStatusText(status);
        }
    }

    // Log status change
    if (previousStatus !== status) {
        const zoneName = state.zones[zoneId].name;
        const logType = status === 'fall' ? 'error' : status === 'warning' ? 'warning' : 'info';
        addLogEntry(logType, `${zoneName} (${zoneId}) 狀態變更: ${getStatusText(status)}`);
    }

    updateGauges();
    updateStats();
}

/**
 * Trigger fall alert
 */
function triggerFallAlert(data) {
    state.currentAlert = data;
    state.detectionCount++;

    const zone = state.zones[data.zone_id];
    const zoneName = zone ? zone.name : data.zone_id;
    const timestamp = new Date().toLocaleTimeString('zh-TW');
    const probability = data.probability ? `${(data.probability * 100).toFixed(1)}%` : '-';

    // Update alert modal
    elements.alertZone.textContent = `${zoneName} (${data.zone_id})`;
    elements.alertTime.textContent = timestamp;
    elements.alertProbability.textContent = probability;

    // Show modal
    elements.alertModal.classList.add('active');

    // Play alert sound
    try {
        elements.alertSound.play();
    } catch (e) {
        console.warn('Could not play alert sound:', e);
    }

    // Update zone status
    updateZoneStatus(data.zone_id, 'fall', { normal: 0, warning: 0, fall: data.probability || 1 });

    // Log alert
    addLogEntry('error', `警報！${zoneName} 偵測到跌倒事件 (機率: ${probability})`);
}

/**
 * Handle detection result from API
 */
function handleDetectionResult(data) {
    if (data.prediction === 'fall' && data.probability > 0.7) {
        triggerFallAlert({
            zone_id: data.zone_id || 'Z08',
            probability: data.probability
        });
    } else if (data.probability > 0.3) {
        updateZoneStatus(data.zone_id || 'Z08', 'warning', {
            normal: 1 - data.probability,
            warning: data.probability * 0.7,
            fall: data.probability * 0.3
        });
    } else {
        updateZoneStatus(data.zone_id || 'Z08', 'normal', {
            normal: 1 - data.probability,
            warning: data.probability * 0.5,
            fall: data.probability * 0.5
        });
    }
}

/**
 * Update radar status
 */
function updateRadarStatus(radarId, online) {
    if (!state.radars[radarId]) return;

    state.radars[radarId].online = online;
    state.radars[radarId].lastPing = Date.now();

    // Update DOM
    const radarCard = document.querySelector(`[data-radar-id="${radarId}"]`);
    if (radarCard) {
        const statusEl = radarCard.querySelector('.radar-card-status');
        if (statusEl) {
            statusEl.className = `radar-card-status ${online ? 'online' : 'offline'}`;
            statusEl.innerHTML = `<span class="status-dot ${online ? 'connected' : 'disconnected'}"></span>${online ? '運行中' : '離線'}`;
        }
    }

    // Log status change
    const radar = state.radars[radarId];
    addLogEntry(online ? 'success' : 'warning', `${radar.name} ${online ? '已上線' : '已離線'}`);
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(status) {
    const dot = elements.connectionStatus.querySelector('.status-dot');
    const text = elements.connectionStatus.querySelector('.status-text');

    dot.className = 'status-dot';

    switch (status) {
        case 'connected':
            dot.classList.add('connected');
            text.textContent = '已連線';
            break;
        case 'connecting':
            dot.classList.add('connecting');
            text.textContent = '連線中...';
            break;
        default:
            dot.classList.add('disconnected');
            text.textContent = '連線中斷';
    }
}

/**
 * Add entry to activity log
 */
function addLogEntry(type, message) {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `
        <span class="log-time">${new Date().toLocaleTimeString('zh-TW')}</span>
        <span class="log-message">${message}</span>
    `;

    elements.activityLog.insertBefore(entry, elements.activityLog.firstChild);

    // Limit log entries
    while (elements.activityLog.children.length > CONFIG.maxLogEntries) {
        elements.activityLog.removeChild(elements.activityLog.lastChild);
    }
}

/**
 * Update gauge displays
 */
function updateGauges() {
    // Calculate average probabilities across all zones
    let totalNormal = 0, totalWarning = 0, totalFall = 0;
    let count = 0;

    Object.values(state.zones).forEach(zone => {
        if (zone.probability) {
            totalNormal += zone.probability.normal || 0;
            totalWarning += zone.probability.warning || 0;
            totalFall += zone.probability.fall || 0;
            count++;
        }
    });

    if (count > 0) {
        const avgNormal = (totalNormal / count) * 100;
        const avgWarning = (totalWarning / count) * 100;
        const avgFall = (totalFall / count) * 100;

        elements.gaugeNormal.style.width = `${avgNormal}%`;
        elements.gaugeWarning.style.width = `${avgWarning}%`;
        elements.gaugeFall.style.width = `${avgFall}%`;

        elements.gaugeNormalValue.textContent = `${avgNormal.toFixed(1)}%`;
        elements.gaugeWarningValue.textContent = `${avgWarning.toFixed(1)}%`;
        elements.gaugeFallValue.textContent = `${avgFall.toFixed(1)}%`;
    }
}

/**
 * Update statistics
 */
function updateStats() {
    // Detection count
    elements.statDetections.textContent = state.detectionCount;

    // Active zones
    const activeZones = Object.values(state.zones).filter(z => z.status !== 'normal').length;
    elements.statActiveZones.textContent = `${activeZones}/15`;

    // Uptime
    const uptime = Date.now() - state.startTime;
    elements.statUptime.textContent = formatDuration(uptime);

    // Last update
    if (state.lastUpdate) {
        elements.statLastUpdate.textContent = state.lastUpdate.toLocaleTimeString('zh-TW');
    }
}

/**
 * Format duration in HH:MM:SS
 */
function formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

/**
 * Get status text in Chinese
 */
function getStatusText(status) {
    const texts = {
        normal: '正常',
        warning: '警告',
        fall: '跌倒'
    };
    return texts[status] || status;
}

/**
 * Render floor plan zones
 */
function renderFloorPlan() {
    elements.floorPlan.innerHTML = '';

    // Render zones
    ZONES.forEach(zone => {
        const zoneEl = document.createElement('div');
        zoneEl.className = 'zone normal';
        zoneEl.dataset.zoneId = zone.id;
        zoneEl.innerHTML = `
            <span class="zone-id">${zone.id}</span>
            <span class="zone-name">${zone.name}</span>
            <span class="zone-status">正常</span>
        `;
        zoneEl.addEventListener('click', () => showZoneDetails(zone.id));
        elements.floorPlan.appendChild(zoneEl);
    });

    // Render radar markers
    RADARS.forEach(radar => {
        const markerEl = document.createElement('div');
        markerEl.className = 'radar-marker';
        markerEl.dataset.radarId = radar.id;
        markerEl.textContent = radar.id;
        markerEl.style.top = radar.position.top;
        markerEl.style.left = radar.position.left;
        markerEl.addEventListener('click', () => showRadarDetails(radar.id));
        elements.floorPlan.appendChild(markerEl);
    });
}

/**
 * Render radar status cards
 */
function renderRadarCards() {
    elements.radarCards.innerHTML = '';

    RADARS.forEach(radar => {
        const cardEl = document.createElement('div');
        cardEl.className = 'radar-card';
        cardEl.dataset.radarId = radar.id;
        cardEl.innerHTML = `
            <div class="radar-card-icon">${radar.id}</div>
            <div class="radar-card-info">
                <div class="radar-card-name">${radar.name}</div>
                <div class="radar-card-location">${radar.location}</div>
            </div>
            <div class="radar-card-status online">
                <span class="status-dot connected"></span>運行中
            </div>
        `;
        elements.radarCards.appendChild(cardEl);
    });
}

/**
 * Show zone details (could open a detail panel)
 */
function showZoneDetails(zoneId) {
    const zone = state.zones[zoneId];
    if (!zone) return;

    addLogEntry('info', `查看區域詳情: ${zone.name} (${zoneId})`);
}

/**
 * Show radar details
 */
function showRadarDetails(radarId) {
    const radar = state.radars[radarId];
    if (!radar) return;

    addLogEntry('info', `查看雷達詳情: ${radar.name} - 覆蓋區域: ${radar.zones.join(', ')}`);
}

/**
 * Update datetime display
 */
function updateDateTime() {
    const now = new Date();
    elements.datetime.textContent = now.toLocaleString('zh-TW', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * Clear activity log
 */
function clearLog() {
    elements.activityLog.innerHTML = '';
    addLogEntry('info', '活動紀錄已清除');
}

/**
 * Dismiss alert
 */
function dismissAlert() {
    elements.alertModal.classList.remove('active');
    state.currentAlert = null;
}

/**
 * Acknowledge alert
 */
function acknowledgeAlert() {
    if (state.currentAlert) {
        const zoneId = state.currentAlert.zone_id;
        addLogEntry('success', `警報已確認處理: ${state.zones[zoneId]?.name || zoneId}`);

        // Reset zone status to warning (being monitored)
        updateZoneStatus(zoneId, 'warning', { normal: 0.5, warning: 0.4, fall: 0.1 });
    }
    dismissAlert();
}

/**
 * Simulate random events for demo purposes (when not connected to real backend)
 */
function simulateEvents() {
    if (state.connected) return;

    // Randomly update zone statuses
    if (Math.random() < 0.1) {
        const zoneIds = Object.keys(state.zones);
        const randomZone = zoneIds[Math.floor(Math.random() * zoneIds.length)];
        const rand = Math.random();

        if (rand < 0.02) {
            // Simulate fall (2% chance)
            triggerFallAlert({
                zone_id: randomZone,
                probability: 0.85 + Math.random() * 0.15
            });
        } else if (rand < 0.15) {
            // Simulate warning (13% chance)
            updateZoneStatus(randomZone, 'warning', {
                normal: 0.4 + Math.random() * 0.2,
                warning: 0.3 + Math.random() * 0.2,
                fall: 0.1 + Math.random() * 0.1
            });
        } else {
            // Return to normal
            updateZoneStatus(randomZone, 'normal', {
                normal: 0.85 + Math.random() * 0.1,
                warning: 0.03 + Math.random() * 0.05,
                fall: 0.01 + Math.random() * 0.02
            });
        }
    }
}

/**
 * Fetch initial status from API
 */
async function fetchInitialStatus() {
    try {
        const response = await fetch(`${CONFIG.apiUrl}/status`);
        if (response.ok) {
            const data = await response.json();
            if (data.zones) {
                Object.entries(data.zones).forEach(([zoneId, zoneData]) => {
                    updateZoneStatus(zoneId, zoneData.status, zoneData.probability);
                });
            }
            if (data.radars) {
                Object.entries(data.radars).forEach(([radarId, radarData]) => {
                    updateRadarStatus(radarId, radarData.online);
                });
            }
            addLogEntry('success', '已從 API 獲取初始狀態');
        }
    } catch (e) {
        console.warn('Could not fetch initial status:', e);
        addLogEntry('warning', '無法獲取初始狀態，使用預設值');
    }
}

/**
 * Initialize dashboard
 */
function init() {
    // Render UI components
    renderFloorPlan();
    renderRadarCards();

    // Setup event listeners
    elements.clearLogBtn.addEventListener('click', clearLog);
    elements.alertAckBtn.addEventListener('click', acknowledgeAlert);
    elements.alertDismissBtn.addEventListener('click', dismissAlert);

    // Close modal on outside click
    elements.alertModal.addEventListener('click', (e) => {
        if (e.target === elements.alertModal) {
            dismissAlert();
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && elements.alertModal.classList.contains('active')) {
            dismissAlert();
        }
    });

    // Start datetime update
    updateDateTime();
    setInterval(updateDateTime, 1000);

    // Start stats update
    setInterval(updateStats, CONFIG.updateInterval);

    // Start simulation for demo
    setInterval(simulateEvents, 2000);

    // Initial log
    addLogEntry('info', '系統啟動');
    addLogEntry('info', '赤土崎多功能館 mmWave 跌倒偵測系統');

    // Try to fetch initial status
    fetchInitialStatus();

    // Connect WebSocket
    connectWebSocket();

    // Update gauges with initial values
    updateGauges();
    updateStats();
}

// Start the application
document.addEventListener('DOMContentLoaded', init);
