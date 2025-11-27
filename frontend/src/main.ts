import * as THREE from "three";

interface Position {
  x: number;
  y: number;
}

interface FallEvent {
  id: string;
  timestamp: number; // unix seconds
  label: string;
  probabilities: Record<string, number>;
  position: Position;
}

interface Stats {
  total: number;
  fall: number;
  normal: number;
  rehab_bad_posture: number;
  chest_abnormal: number;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const WS_BASE = import.meta.env.VITE_WS_BASE ?? "ws://localhost:8000";

// Sound notification state
let soundEnabled = false;
let audioContext: AudioContext | null = null;

function initAudioContext(): void {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
}

function playAlertSound(): void {
  if (!soundEnabled || !audioContext) return;

  try {
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.setValueAtTime(880, audioContext.currentTime);
    oscillator.type = "sine";

    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);

    // Second beep
    setTimeout(() => {
      if (!audioContext) return;
      const osc2 = audioContext.createOscillator();
      const gain2 = audioContext.createGain();
      osc2.connect(gain2);
      gain2.connect(audioContext.destination);
      osc2.frequency.setValueAtTime(1100, audioContext.currentTime);
      osc2.type = "sine";
      gain2.gain.setValueAtTime(0.3, audioContext.currentTime);
      gain2.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
      osc2.start(audioContext.currentTime);
      osc2.stop(audioContext.currentTime + 0.3);
    }, 200);
  } catch (err) {
    console.warn("Audio playback failed:", err);
  }
}

function triggerVisualAlert(): void {
  const alertOverlay = document.getElementById("alert-overlay");
  if (alertOverlay) {
    alertOverlay.classList.add("active");
    setTimeout(() => alertOverlay.classList.remove("active"), 2000);
  }

  document.body.classList.add("fall-alert-active");
  setTimeout(() => document.body.classList.remove("fall-alert-active"), 2000);
}

const hallWidth = 20;
const hallHeight = 10;

function colorForLabel(label: string): number {
  switch (label) {
    case "fall":
      return 0xef4444; // red
    case "rehab_bad_posture":
      return 0xf97316; // orange
    case "chest_abnormal":
      return 0x22d3ee; // cyan
    case "normal":
    default:
      return 0x22c55e; // green
  }
}

function sizeForLabel(label: string): number {
  switch (label) {
    case "fall":
      return 0.5;
    case "rehab_bad_posture":
    case "chest_abnormal":
      return 0.35;
    case "normal":
    default:
      return 0.25;
  }
}

interface EventMeshData {
  mesh: THREE.Mesh;
  ring?: THREE.Mesh;
  label: string;
  createdAt: number;
}

function setupScene(canvas: HTMLCanvasElement) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(canvas.clientWidth || 800, canvas.clientHeight || 450, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101018);

  const aspect = (canvas.clientWidth || 800) / (canvas.clientHeight || 450);
  const viewWidth = hallWidth * 1.3;
  const viewHeight = viewWidth / aspect;

  const camera = new THREE.OrthographicCamera(
    -viewWidth / 2,
    viewWidth / 2,
    viewHeight / 2,
    -viewHeight / 2,
    0.1,
    100
  );
  camera.position.set(0, 20, 0);
  camera.up.set(0, 0, -1);
  camera.lookAt(0, 0, 0);

  const floorGeom = new THREE.PlaneGeometry(hallWidth, hallHeight);
  const floorMat = new THREE.MeshBasicMaterial({ color: 0x1f2937 });
  const floor = new THREE.Mesh(floorGeom, floorMat);
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const grid = new THREE.GridHelper(hallWidth, hallWidth, 0x4b5563, 0x374151);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const light = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(light);

  const eventMeshes = new Map<string, EventMeshData>();
  let animationTime = 0;

  function upsertEventMesh(ev: FallEvent, isNew = false) {
    let data = eventMeshes.get(ev.id);
    const color = colorForLabel(ev.label);
    const size = sizeForLabel(ev.label);

    if (!data) {
      const geom = new THREE.CircleGeometry(size, 24);
      const mat = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.rotation.x = -Math.PI / 2;
      scene.add(mesh);

      data = { mesh, label: ev.label, createdAt: Date.now() };

      // Add pulsing ring for fall events
      if (ev.label === "fall") {
        const ringGeom = new THREE.RingGeometry(size * 1.2, size * 1.5, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xef4444,
          transparent: true,
          opacity: 0.8,
        });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.rotation.x = -Math.PI / 2;
        scene.add(ring);
        data.ring = ring;
      }

      eventMeshes.set(ev.id, data);

      // Trigger alerts for new fall events
      if (isNew && ev.label === "fall") {
        triggerVisualAlert();
        playAlertSound();
      }
    } else {
      (data.mesh.material as THREE.MeshBasicMaterial).color.set(color);
      data.mesh.geometry.dispose();
      data.mesh.geometry = new THREE.CircleGeometry(size, 24);
      data.label = ev.label;

      // Add or remove ring based on label change
      if (ev.label === "fall" && !data.ring) {
        const ringGeom = new THREE.RingGeometry(size * 1.2, size * 1.5, 32);
        const ringMat = new THREE.MeshBasicMaterial({
          color: 0xef4444,
          transparent: true,
          opacity: 0.8,
        });
        const ring = new THREE.Mesh(ringGeom, ringMat);
        ring.rotation.x = -Math.PI / 2;
        scene.add(ring);
        data.ring = ring;

        if (isNew) {
          triggerVisualAlert();
          playAlertSound();
        }
      } else if (ev.label !== "fall" && data.ring) {
        scene.remove(data.ring);
        data.ring.geometry.dispose();
        (data.ring.material as THREE.Material).dispose();
        data.ring = undefined;
      }
    }

    const clampedX = Math.max(-hallWidth / 2, Math.min(hallWidth / 2, ev.position.x));
    const clampedY = Math.max(-hallHeight / 2, Math.min(hallHeight / 2, ev.position.y));

    data.mesh.position.set(clampedX, 0.01, clampedY);
    if (data.ring) {
      data.ring.position.set(clampedX, 0.02, clampedY);
    }
  }

  function render() {
    renderer.render(scene, camera);
  }

  function resizeRendererToDisplaySize() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
      renderer.setSize(width, height, false);
    }
  }

  function animate() {
    animationTime += 0.05;
    resizeRendererToDisplaySize();

    // Animate fall event rings (pulsing effect)
    eventMeshes.forEach((data) => {
      if (data.ring && data.label === "fall") {
        const scale = 1 + Math.sin(animationTime * 3) * 0.3;
        data.ring.scale.set(scale, scale, 1);
        (data.ring.material as THREE.MeshBasicMaterial).opacity =
          0.4 + Math.sin(animationTime * 3) * 0.4;
      }
    });

    render();
    requestAnimationFrame(animate);
  }
  animate();

  return { upsertEventMesh };
}

async function fetchRecentEvents(): Promise<FallEvent[]> {
  try {
    const res = await fetch(`${API_BASE}/events/recent`);
    if (!res.ok) {
      console.warn("Failed to fetch events:", res.status);
      return [];
    }
    const data = await res.json();
    return data.events ?? [];
  } catch (err) {
    console.warn("Error fetching events", err);
    return [];
  }
}

function mostLikelyProb(ev: FallEvent): number {
  const values = Object.values(ev.probabilities ?? {});
  if (!values.length) return 0;
  return Math.max(...values);
}

function updateTimeline(events: FallEvent[]) {
  const ul = document.getElementById("event-timeline");
  if (!ul) return;
  ul.innerHTML = "";

  const sorted = [...events].sort((a, b) => b.timestamp - a.timestamp);

  for (const ev of sorted) {
    const li = document.createElement("li");
    li.className = "event-item";

    const time = new Date(ev.timestamp * 1000).toLocaleTimeString();
    const prob = (mostLikelyProb(ev) * 100).toFixed(1);
    li.innerHTML = `<strong>[${time}]</strong> ${ev.label.toUpperCase()} · p*=${prob}% · (x=${ev.position.x.toFixed(
      1
    )}, y=${ev.position.y.toFixed(1)})`;

    li.dataset.label = ev.label;
    if (ev.label === "fall") {
      li.classList.add("event-fall");
    } else if (ev.label === "rehab_bad_posture") {
      li.classList.add("event-rehab");
    } else if (ev.label === "chest_abnormal") {
      li.classList.add("event-chest");
    } else {
      li.classList.add("event-normal");
    }
    ul.appendChild(li);
  }
}

function calculateStats(events: FallEvent[]): Stats {
  const stats: Stats = {
    total: events.length,
    fall: 0,
    normal: 0,
    rehab_bad_posture: 0,
    chest_abnormal: 0,
  };

  for (const ev of events) {
    if (ev.label === "fall") stats.fall++;
    else if (ev.label === "normal") stats.normal++;
    else if (ev.label === "rehab_bad_posture") stats.rehab_bad_posture++;
    else if (ev.label === "chest_abnormal") stats.chest_abnormal++;
  }

  return stats;
}

function updateStatsPanel(stats: Stats): void {
  const panel = document.getElementById("stats-panel");
  if (!panel) return;

  panel.innerHTML = `
    <div class="stat-item stat-total">
      <span class="stat-value">${stats.total}</span>
      <span class="stat-label">Total Events</span>
    </div>
    <div class="stat-item stat-fall">
      <span class="stat-value">${stats.fall}</span>
      <span class="stat-label">Falls</span>
    </div>
    <div class="stat-item stat-normal">
      <span class="stat-value">${stats.normal}</span>
      <span class="stat-label">Normal</span>
    </div>
    <div class="stat-item stat-rehab">
      <span class="stat-value">${stats.rehab_bad_posture}</span>
      <span class="stat-label">Bad Posture</span>
    </div>
    <div class="stat-item stat-chest">
      <span class="stat-value">${stats.chest_abnormal}</span>
      <span class="stat-label">Chest Abnormal</span>
    </div>
  `;
}

function updateConnectionStatus(connected: boolean): void {
  const indicator = document.getElementById("connection-status");
  if (!indicator) return;

  indicator.className = connected ? "status-connected" : "status-disconnected";
  indicator.textContent = connected ? "Live" : "Disconnected";
}

function setupSoundToggle(): void {
  const toggle = document.getElementById("sound-toggle");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    initAudioContext();
    soundEnabled = !soundEnabled;
    toggle.textContent = soundEnabled ? "Sound: ON" : "Sound: OFF";
    toggle.classList.toggle("active", soundEnabled);

    // Resume audio context if suspended (browser autoplay policy)
    if (soundEnabled && audioContext?.state === "suspended") {
      audioContext.resume();
    }
  });
}

class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectTimeout: number | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private onEventCallback: ((event: FallEvent) => void) | null = null;

  constructor(private url: string) {}

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log("WebSocket connected");
        updateConnectionStatus(true);
        this.reconnectDelay = 1000;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "event" && data.event) {
            this.onEventCallback?.(data.event as FallEvent);
          } else if (data.type === "events" && Array.isArray(data.events)) {
            for (const ev of data.events) {
              this.onEventCallback?.(ev as FallEvent);
            }
          }
        } catch (err) {
          console.warn("Failed to parse WebSocket message:", err);
        }
      };

      this.ws.onclose = () => {
        console.log("WebSocket disconnected");
        updateConnectionStatus(false);
        this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.warn("WebSocket error:", err);
        updateConnectionStatus(false);
      };
    } catch (err) {
      console.warn("Failed to create WebSocket:", err);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) return;

    this.reconnectTimeout = window.setTimeout(() => {
      this.reconnectTimeout = null;
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
      this.connect();
    }, this.reconnectDelay);
  }

  onEvent(callback: (event: FallEvent) => void): void {
    this.onEventCallback = callback;
  }

  disconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    this.ws?.close();
    this.ws = null;
  }
}

async function main() {
  const canvas = document.getElementById("scene-canvas") as HTMLCanvasElement | null;
  if (!canvas) {
    console.error("Missing #scene-canvas element");
    return;
  }

  const { upsertEventMesh } = setupScene(canvas);
  const cachedEvents = new Map<string, FallEvent>();

  function addOrUpdateEvent(ev: FallEvent, isNew = false): void {
    const existing = cachedEvents.get(ev.id);
    const isActuallyNew = !existing;
    cachedEvents.set(ev.id, ev);
    upsertEventMesh(ev, isNew && isActuallyNew);

    // Update UI
    const eventsArray = Array.from(cachedEvents.values());
    updateTimeline(eventsArray);
    updateStatsPanel(calculateStats(eventsArray));
  }

  // Setup sound toggle button
  setupSoundToggle();

  // Initial fetch via HTTP
  async function refresh() {
    const events = await fetchRecentEvents();
    for (const ev of events) {
      addOrUpdateEvent(ev, false);
    }
  }

  await refresh();

  // Setup WebSocket for real-time updates
  const wsManager = new WebSocketManager(`${WS_BASE}/ws/events`);
  wsManager.onEvent((ev) => {
    addOrUpdateEvent(ev, true);
  });
  wsManager.connect();

  // Fallback polling in case WebSocket is unavailable
  setInterval(refresh, 10000);
}

window.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("scene-canvas") as HTMLCanvasElement | null;
  if (canvas) {
    const parent = canvas.parentElement;
    if (parent) {
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    }
  }
  void main();
});
