import * as THREE from "three";

interface Position {
  x: number;
  y: number;
}

interface FallEvent {
  id: string;
  timestamp: number; // unix seconds
  label: string;
  prob_fall: number;
  prob_normal: number;
  position: Position;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const hallWidth = 20;
const hallHeight = 10;

function setupScene(canvas: HTMLCanvasElement) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(canvas.clientWidth || 800, canvas.clientHeight || 450, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101018);

  // Orthographic camera for top‑down 2D style view
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

  // Simple hall floor
  const floorGeom = new THREE.PlaneGeometry(hallWidth, hallHeight);
  const floorMat = new THREE.MeshBasicMaterial({ color: 0x1f2937 });
  const floor = new THREE.Mesh(floorGeom, floorMat);
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  // Grid / lines for reference
  const grid = new THREE.GridHelper(hallWidth, hallWidth, 0x4b5563, 0x374151);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  // Light (for any future 3D avatars)
  const light = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(light);

  const eventMeshes = new Map<string, THREE.Mesh>();

  function upsertEventMesh(ev: FallEvent) {
    let mesh = eventMeshes.get(ev.id);
    const color = ev.label === "fall" ? 0xef4444 : 0x22c55e;
    if (!mesh) {
      const geom = new THREE.CircleGeometry(0.3, 24);
      const mat = new THREE.MeshBasicMaterial({ color });
      mesh = new THREE.Mesh(geom, mat);
      mesh.rotation.x = -Math.PI / 2;
      scene.add(mesh);
      eventMeshes.set(ev.id, mesh);
    } else {
      (mesh.material as THREE.MeshBasicMaterial).color.set(color);
    }

    const clampedX = Math.max(-hallWidth / 2, Math.min(hallWidth / 2, ev.position.x));
    const clampedY = Math.max(-hallHeight / 2, Math.min(hallHeight / 2, ev.position.y));

    mesh.position.set(clampedX, 0.01, clampedY);
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
    resizeRendererToDisplaySize();
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

function updateTimeline(events: FallEvent[]) {
  const ul = document.getElementById("event-timeline");
  if (!ul) return;
  ul.innerHTML = "";

  const sorted = [...events].sort((a, b) => b.timestamp - a.timestamp);

  for (const ev of sorted) {
    const li = document.createElement("li");
    li.className = "event-item";

    const time = new Date(ev.timestamp * 1000).toLocaleTimeString();
    const prob = (ev.prob_fall * 100).toFixed(1);
    li.innerHTML = `<strong>[${time}]</strong> ${ev.label.toUpperCase()} · p(fall)=${prob}% · (x=${ev.position.x.toFixed(
      1
    )}, y=${ev.position.y.toFixed(1)})`;

    if (ev.label === "fall") {
      li.classList.add("event-fall");
    } else {
      li.classList.add("event-normal");
    }
    ul.appendChild(li);
  }
}

async function main() {
  const canvas = document.getElementById("scene-canvas") as HTMLCanvasElement | null;
  if (!canvas) {
    console.error("Missing #scene-canvas element");
    return;
  }

  const { upsertEventMesh } = setupScene(canvas);
  const seen = new Set<string>();
  let cachedEvents: FallEvent[] = [];

  async function refresh() {
    const events = await fetchRecentEvents();
    if (!events.length) return;

    cachedEvents = events;
    for (const ev of events) {
      if (!seen.has(ev.id)) {
        seen.add(ev.id);
      }
      upsertEventMesh(ev);
    }
    updateTimeline(cachedEvents);
  }

  await refresh();
  setInterval(refresh, 3000);
}

window.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("scene-canvas") as HTMLCanvasElement | null;
  if (canvas) {
    // Expand canvas to fill its container
    const parent = canvas.parentElement;
    if (parent) {
      canvas.width = parent.clientWidth;
      canvas.height = parent.clientHeight;
    }
  }
  void main();
});
