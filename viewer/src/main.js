/**
 * main.js — 4D Scene Viewer
 *
 * Supports two modes:
 *   1. Demo scenes (.ksplat): high-quality pre-trained scenes for navigation demo
 *   2. My Scene (manifest.json): pipeline output with 4D timeline
 */

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { Timeline } from "./timeline.js";
import { Controls } from "./controls.js";

const MANIFEST_URL = "./manifest.json";

const DEMO_SCENES = [
  {
    id:    "bonsai",
    label: "Bonsai",
    path:  "./demo_scenes/bonsai.ksplat",
    cameraPosition: [0.45, -0.3, -1.6],
    cameraLookAt:   [0, 0.1, 0],
  },
  {
    id:    "truck",
    label: "Truck",
    path:  "./demo_scenes/truck.ksplat",
    cameraPosition: [1.5, -0.6, -3.5],
    cameraLookAt:   [0, 0, 0],
  },
];

// ── State ──────────────────────────────────────────────────────────────────────

let viewer         = null;
let manifest       = null;
let timeline       = null;
let controls       = null;
let viewerStarted  = false;
let activeSceneId  = null;

let currentDisplayedFrame = -1;
let isDisplaying          = false;
const frameCache = new Map();

// ── Loading overlay ────────────────────────────────────────────────────────────

function setLoading(visible, text = "Loading…", progress = null) {
  const overlay = document.getElementById("loading-overlay");
  const textEl  = document.getElementById("loading-text");
  const bar     = document.getElementById("progress-bar");
  if (overlay) overlay.classList.toggle("hidden", !visible);
  if (textEl)  textEl.textContent = text;
  if (bar && progress !== null) bar.style.width = `${Math.round(progress * 100)}%`;
}

function showError(msg) {
  const overlay = document.getElementById("loading-overlay");
  const textEl  = document.getElementById("loading-text");
  const bar     = document.getElementById("progress-bar");
  if (overlay) overlay.classList.remove("hidden");
  if (textEl)  { textEl.textContent = `⚠ ${msg}`; textEl.style.color = "#f08080"; }
  if (bar)     bar.style.background = "#c04040";
  console.error(msg);
}

// ── Viewer lifecycle ───────────────────────────────────────────────────────────

function destroyViewer() {
  if (!viewer) return;
  try { viewer.stop?.(); } catch (_) {}
  try { viewer.dispose?.(); } catch (_) {}
  viewer = null;
  viewerStarted = false;
  currentDisplayedFrame = -1;
  frameCache.clear();
  timeline = null;
  controls = null;
}

function createViewer(cameraPosition = null, cameraLookAt = null) {
  const container = document.getElementById("canvas-container");
  // Remove any leftover canvas
  container.querySelectorAll("canvas").forEach(c => c.remove());

  const opts = {
    rootElement:            container,
    selfDrivenMode:         true,
    useBuiltInControls:     true,
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort:     false,
    logLevel:               GaussianSplats3D.LogLevel.Warning,
    cameraUp:               [0, -1, 0],
  };
  if (cameraPosition) opts.initialCameraPosition = cameraPosition;
  if (cameraLookAt)   opts.initialCameraLookAt   = cameraLookAt;

  viewer = new GaussianSplats3D.Viewer(opts);
}

// ── Splat / ksplat loading ─────────────────────────────────────────────────────

async function addScene(path, format = null) {
  const opts = {
    splatAlphaRemovalThreshold: 10,
    showLoadingUI:              false,
    position:                   [0, 0, 0],
    rotation:                   [0, 0, 0, 1],
    scale:                      [1, 1, 1],
  };
  if (format) opts.format = format;
  await viewer.addSplatScene(path, opts);
}

// ── Prefetch / display (4D timeline mode) ─────────────────────────────────────

async function prefetchFrame(idx) {
  if (!manifest || frameCache.has(idx) || idx < 0 || idx >= manifest.n_frames) return;
  try {
    const res = await fetch(`./${manifest.files[idx].file}`);
    if (res.ok) frameCache.set(idx, await res.arrayBuffer());
  } catch (_) {}
}

function prefetchNeighbours(center) {
  for (let d = 1; d <= 3; d++) {
    prefetchFrame(center + d);
    prefetchFrame(center - d);
  }
}

async function displayFrame(frameIdx) {
  if (frameIdx === currentDisplayedFrame || isDisplaying) return true;
  isDisplaying = true;
  try {
    if (frameCache.has(frameIdx)) {
      const blob = new Blob([frameCache.get(frameIdx)], { type: "application/octet-stream" });
      const url  = URL.createObjectURL(blob);
      try   { await addScene(url, GaussianSplats3D.SceneFormat.Splat); }
      finally { URL.revokeObjectURL(url); }
    } else {
      await addScene(`./${manifest.files[frameIdx].file}`, GaussianSplats3D.SceneFormat.Splat);
    }

    const nScenes = viewer.splatMesh?.scenes?.length ?? 0;
    for (let i = nScenes - 2; i >= 0; i--) {
      try { await viewer.removeSplatScene(i, false); } catch (_) {}
    }

    if (!viewerStarted) {
      viewer.start();
      viewerStarted = true;
      requestAnimationFrame(renderLoop);
    }

    currentDisplayedFrame = frameIdx;
    prefetchNeighbours(frameIdx);
    return true;
  } catch (err) {
    showError(`Frame ${frameIdx} failed: ${err.message}`);
    return false;
  } finally {
    isDisplaying = false;
  }
}

// ── Render loop ────────────────────────────────────────────────────────────────

function renderLoop() {
  requestAnimationFrame(renderLoop);
  timeline?.tick(performance.now() / 1000 - (renderLoop._t ?? 0));
  renderLoop._t = performance.now() / 1000;
  controls?.tickFps();
}

// ── Scene switching ────────────────────────────────────────────────────────────

function setActiveBtn(id) {
  document.querySelectorAll(".scene-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.scene === id);
  });
  // Hide timeline for demo scenes, show for my_scene
  const timelineEl = document.getElementById("timeline-container");
  if (timelineEl) timelineEl.style.display = (id === "my_scene") ? "" : "none";
}

async function loadDemoScene(scene) {
  if (activeSceneId === scene.id) return;
  activeSceneId = scene.id;
  setActiveBtn(scene.id);
  destroyViewer();
  setLoading(true, `Loading ${scene.label}…`, 0);

  createViewer(scene.cameraPosition, scene.cameraLookAt);

  try {
    await addScene(scene.path);
    viewer.start();
    viewerStarted = true;
    requestAnimationFrame(renderLoop);
    setLoading(false);
  } catch (err) {
    showError(`Could not load ${scene.label}: ${err.message}`);
  }
}

async function loadMyScene() {
  if (activeSceneId === "my_scene") return;
  activeSceneId = "my_scene";
  setActiveBtn("my_scene");
  destroyViewer();
  setLoading(true, "Loading pipeline output…", 0);

  // Camera at mean training position, looking at scene center
  createViewer([-0.112, -0.002, 0.088], [-0.022, 0.004, 0.26]);

  try {
    const m = await (await fetch(MANIFEST_URL)).json();
    manifest = m;
    const ok = await displayFrame(0);
    if (!ok) return;

    timeline = new Timeline({
      nFrames:    m.n_frames,
      timestamps: m.timestamps ?? Array.from({ length: m.n_frames }, (_, i) => i),
      onSeek:     (f) => displayFrame(f),
    });

    if (viewer.camera) {
      controls = new Controls({
        camera:        viewer.camera,
        renderer:      viewer.renderer,
        orbitControls: viewer.orbitControls ?? null,
      });
      controls.saveHomeCamera();
    }

    setLoading(false);
  } catch (err) {
    showError(`Pipeline output failed: ${err.message}`);
  }
}

// ── Build scene picker UI ──────────────────────────────────────────────────────

function buildScenePicker() {
  const bar = document.getElementById("scene-picker");
  if (!bar) return;

  DEMO_SCENES.forEach(scene => {
    const btn = document.createElement("button");
    btn.className    = "ctrl-btn scene-btn";
    btn.dataset.scene = scene.id;
    btn.textContent  = scene.label;
    btn.addEventListener("click", () => loadDemoScene(scene));
    bar.appendChild(btn);
  });

  const myBtn = document.createElement("button");
  myBtn.className    = "ctrl-btn scene-btn";
  myBtn.dataset.scene = "my_scene";
  myBtn.textContent  = "My Scene";
  myBtn.addEventListener("click", loadMyScene);
  bar.appendChild(myBtn);
}

// ── Boot ───────────────────────────────────────────────────────────────────────

async function boot() {
  buildScenePicker();
  // Default: load bonsai first (best looking demo)
  await loadDemoScene(DEMO_SCENES[0]);
}

boot();
