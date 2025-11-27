/**
 * mmWave Fall Detection Dashboard
 * Real-time monitoring interface
 */

// Configuration
const CONFIG = {
    API_BASE_URL: 'https://analyze-wales-porter-tip.trycloudflare.com',
    WS_URL: 'wss://analyze-wales-porter-tip.trycloudflare.com/ws/events',
    MAX_EVENTS: 20,
    STATS_POLL_INTERVAL: 5000,
    HEALTH_CHECK_INTERVAL: 10000,
    ALERT_DURATION: 5000,
    MAP_WIDTH: 10,  // meters
    MAP_HEIGHT: 8   // meters
};

// State
const state = {
    connected: false,
    wsConnected: false,
    events: [],
    fallPositions: [],
    normalPositions: []
};

// DOM Elements
const elements = {
    statusIndicator: document.getElementById('statusIndicator'),
    statusText: document.getElementById('statusText'),
    totalEvents: document.getElementById('totalEvents'),
    fallCount: document.getElementById('fallCount'),
    normalCount: document.getElementById('normalCount'),
    wsConnections: document.getElementById('wsConnections'),
    eventList: document.getElementById('eventList'),
    floorMap: document.getElementById('floorMap'),
    fallAlert: document.getElementById('fallAlert'),
    alertDetails: document.getElementById('alertDetails')
};

// Canvas context
const ctx = elements.floorMap.getContext('2d');

/**
 * Initialize the dashboard
 */
function init() {
    console.log('Initializing mmWave Fall Detection Dashboard...');

    // Set canvas size based on container
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Start health check polling
    checkHealth();
    setInterval(checkHealth, CONFIG.HEALTH_CHECK_INTERVAL);

    // Start statistics polling
    fetchStats();
    setInterval(fetchStats, CONFIG.STATS_POLL_INTERVAL);

    // Fetch recent events
    fetchRecentEvents();

    // Connect WebSocket
    connectWebSocket();

    // Initial render
    renderFloorMap();
}

/**
 * Resize canvas to maintain aspect ratio
 */
function resizeCanvas() {
    const container = elements.floorMap.parentElement;
    const width = Math.min(container.clientWidth - 40, 500);
    const height = width * (CONFIG.MAP_HEIGHT / CONFIG.MAP_WIDTH);
    elements.floorMap.width = width;
    elements.floorMap.height = height;
    renderFloorMap();
}

/**
 * Check API health
 */
async function checkHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        if (response.ok) {
            setConnectionStatus(true, 'API Connected');
        } else {
            setConnectionStatus(false, 'API Error');
        }
    } catch (error) {
        setConnectionStatus(false, 'API Disconnected');
        console.error('Health check failed:', error);
    }
}

/**
 * Fetch statistics from API
 */
async function fetchStats() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/stats`);
        if (response.ok) {
            const stats = await response.json();
            updateStats(stats);
        }
    } catch (error) {
        console.error('Failed to fetch stats:', error);
    }
}

/**
 * Fetch recent events from API
 */
async function fetchRecentEvents() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/events/recent?limit=${CONFIG.MAX_EVENTS}`);
        if (response.ok) {
            const events = await response.json();
            state.events = events.reverse();
            renderEventList();
            updatePositionsFromEvents(events);
        }
    } catch (error) {
        console.error('Failed to fetch recent events:', error);
    }
}

/**
 * Update statistics display
 */
function updateStats(stats) {
    elements.totalEvents.textContent = formatNumber(stats.total_events || 0);
    elements.fallCount.textContent = formatNumber(stats.label_counts?.fall || 0);
    elements.normalCount.textContent = formatNumber(stats.label_counts?.normal || 0);
    elements.wsConnections.textContent = stats.websocket_connections || 0;
}

/**
 * Format large numbers
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

/**
 * Set connection status
 */
function setConnectionStatus(connected, text) {
    state.connected = connected;
    elements.statusIndicator.className = 'status-indicator ' + (connected ? 'connected' : 'disconnected');
    elements.statusText.textContent = text;
}

/**
 * Connect to WebSocket for real-time updates
 */
function connectWebSocket() {
    console.log('Connecting to WebSocket...');

    const ws = new WebSocket(CONFIG.WS_URL);

    ws.onopen = () => {
        console.log('WebSocket connected');
        state.wsConnected = true;
        setConnectionStatus(true, 'Live Connected');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketEvent(data);
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        state.wsConnected = false;
        setConnectionStatus(state.connected, 'API Only');

        // Reconnect after delay
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

/**
 * Handle incoming WebSocket event
 */
function handleWebSocketEvent(data) {
    // Handle different event types
    if (data.type === 'prediction' || data.label) {
        const event = {
            id: data.id || Date.now(),
            timestamp: data.timestamp || new Date().toISOString(),
            label: data.label || data.prediction,
            confidence: data.confidence || data.probability || 0,
            position: data.position || generateRandomPosition()
        };

        addEvent(event);

        // Show alert for fall events
        if (event.label === 'fall') {
            showFallAlert(event);
        }
    } else if (data.type === 'stats') {
        updateStats(data);
    }
}

/**
 * Add new event to the list
 */
function addEvent(event) {
    // Add to beginning of array
    state.events.unshift(event);

    // Keep only max events
    if (state.events.length > CONFIG.MAX_EVENTS) {
        state.events.pop();
    }

    // Update positions
    if (event.position) {
        if (event.label === 'fall') {
            state.fallPositions.push({
                x: event.position.x,
                y: event.position.y,
                timestamp: Date.now()
            });
            // Keep only last 10 fall positions
            if (state.fallPositions.length > 10) {
                state.fallPositions.shift();
            }
        } else {
            state.normalPositions.push({
                x: event.position.x,
                y: event.position.y,
                timestamp: Date.now()
            });
            // Keep only last 20 normal positions
            if (state.normalPositions.length > 20) {
                state.normalPositions.shift();
            }
        }
    }

    // Render updates
    renderEventList();
    renderFloorMap();
}

/**
 * Render the event list
 */
function renderEventList() {
    if (state.events.length === 0) {
        elements.eventList.innerHTML = '<div class="event-placeholder">Waiting for events...</div>';
        return;
    }

    elements.eventList.innerHTML = state.events.map(event => {
        const isFall = event.label === 'fall';
        const time = formatTime(event.timestamp);
        const confidence = (event.confidence * 100).toFixed(1);

        return `
            <div class="event-item ${isFall ? 'fall' : ''}">
                <span class="event-type">${event.label}</span>
                <span class="event-details">
                    Confidence: ${confidence}%
                    ${event.position ? ` | Pos: (${event.position.x.toFixed(1)}, ${event.position.y.toFixed(1)})` : ''}
                </span>
                <span class="event-time">${time}</span>
            </div>
        `;
    }).join('');
}

/**
 * Format timestamp to readable time
 */
function formatTime(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

/**
 * Render the floor map with positions
 */
function renderFloorMap() {
    const canvas = elements.floorMap;
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#21262d';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;

    const gridSize = width / 10;
    for (let x = 0; x <= width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
    for (let y = 0; y <= height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    // Draw room outline
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.strokeRect(10, 10, width - 20, height - 20);

    // Draw radar position (center top)
    ctx.fillStyle = '#58a6ff';
    ctx.beginPath();
    ctx.arc(width / 2, 25, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#f0f6fc';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('RADAR', width / 2, 45);

    // Draw normal positions (green)
    state.normalPositions.forEach(pos => {
        const x = (pos.x / CONFIG.MAP_WIDTH) * (width - 40) + 20;
        const y = (pos.y / CONFIG.MAP_HEIGHT) * (height - 60) + 50;

        ctx.fillStyle = 'rgba(63, 185, 80, 0.6)';
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();
    });

    // Draw fall positions (red with glow)
    state.fallPositions.forEach((pos, index) => {
        const x = (pos.x / CONFIG.MAP_WIDTH) * (width - 40) + 20;
        const y = (pos.y / CONFIG.MAP_HEIGHT) * (height - 60) + 50;

        // Glow effect
        ctx.shadowColor = '#f85149';
        ctx.shadowBlur = 15;

        ctx.fillStyle = '#f85149';
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();

        // Reset shadow
        ctx.shadowBlur = 0;

        // Add pulse animation for recent falls
        const age = Date.now() - pos.timestamp;
        if (age < 5000) {
            const opacity = 1 - (age / 5000);
            ctx.strokeStyle = `rgba(248, 81, 73, ${opacity})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, 15 + (age / 200), 0, Math.PI * 2);
            ctx.stroke();
        }
    });
}

/**
 * Show fall alert overlay
 */
function showFallAlert(event) {
    const time = formatTime(event.timestamp);
    const position = event.position ?
        `Position: (${event.position.x.toFixed(1)}, ${event.position.y.toFixed(1)})` : '';

    elements.alertDetails.textContent = `${time} | Confidence: ${(event.confidence * 100).toFixed(1)}% | ${position}`;
    elements.fallAlert.classList.add('active');

    // Auto-hide after duration
    setTimeout(() => {
        elements.fallAlert.classList.remove('active');
    }, CONFIG.ALERT_DURATION);

    // Click to dismiss
    elements.fallAlert.onclick = () => {
        elements.fallAlert.classList.remove('active');
    };
}

/**
 * Generate random position for demo purposes
 */
function generateRandomPosition() {
    return {
        x: Math.random() * CONFIG.MAP_WIDTH,
        y: Math.random() * CONFIG.MAP_HEIGHT
    };
}

/**
 * Update positions from historical events
 */
function updatePositionsFromEvents(events) {
    state.fallPositions = [];
    state.normalPositions = [];

    events.forEach(event => {
        const position = event.position || generateRandomPosition();
        if (event.label === 'fall') {
            state.fallPositions.push({
                x: position.x,
                y: position.y,
                timestamp: new Date(event.timestamp).getTime()
            });
        } else {
            state.normalPositions.push({
                x: position.x,
                y: position.y,
                timestamp: new Date(event.timestamp).getTime()
            });
        }
    });

    renderFloorMap();
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
