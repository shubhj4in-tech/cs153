/**
 * main.js — 4D Scene Viewer
 *
 * Strategy: cache each frame's raw .splat bytes as an ArrayBuffer.
 * On frame switch, remove the current GaussianSplats3D scene, create a
 * blob URL from the cached bytes, and load the new scene.  This avoids
 * the "all scenes merged" problem from addSplatScene accumulation.
 */

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { Timeline } from "./timeline.js";
import { Controls } from "./controls.js";

// ── Config ──────────────────────────────────────────────────────────────────

const MANIFEST_URL   = "./manifest.json";
const PRELOAD_RADIUS = 2;  // how many frames to preload ahead/behind

// ── State ───────────────────────────────────────────────────────────────────

let viewer   = null;
let manifest = null;
let timeline = null;
let controls = null;

// Raw byte cache: frameIndex → ArrayBuffer
const frameBuffers = new Map();
let   currentDisplayedFrame = -1;
let   isDisplaying = false;      // guard against concurrent displayFrame calls

// ── Loading overlay ──────────────────────────────────────────────────────────

function setLoading(visible, text = "Loading…", progress = null) {
  const overlay = document.getElementById("loading-overlay");
  const textEl  = document.getElementById("loading-text");
  const bar     = document.getElementById("progress-bar");
  if (overlay) overlay.classList.toggle("hidden", !visible);
  if (textEl)  textEl.textContent = text;
  if (bar && progress !== null) bar.style.width = `${Math.round(progress * 100)}%`;
}

// ── Manifest ─────────────────────────────────────────────────────────────────

async function loadManifest(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load manifest: ${res.status} ${url}`);
  return res.json();
}

// ── Viewer setup ─────────────────────────────────────────────────────────────

function createViewer() {
  const container = document.getElementById("canvas-container");
  viewer = new GaussianSplats3D.Viewer({
    rootElement:           container,
    selfDrivenMode:        false,
    useBuiltInControls:    true,
    initialCameraPosition: [0, -3, 5],
    initialCameraLookAt:   [0, 0, 0],
    cameraUp:              [0, 1, 0],
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort:    true,
    logLevel:              GaussianSplats3D.LogLevel.None,
  });
  return viewer;
}

// ── Byte fetching (preload without parsing) ──────────────────────────────────

async function fetchFrame(frameIdx) {
  if (frameBuffers.has(frameIdx)) return;
  const file = manifest.files[frameIdx];
  const res  = await fetch(`./${file.file}`);
  if (!res.ok) throw new Error(`Fetch failed for frame ${frameIdx}: ${res.status}`);
  frameBuffers.set(frameIdx, await res.arrayBuffer());
}

// ── Scene management ─────────────────────────────────────────────────────────

async function clearAllScenes() {
  if (!viewer) return;
  // removeSplatScene(index) exists in GaussianSplats3D >= 0.3.x.
  // Remove from highest index downward to avoid index shifting.
  const nScenes = viewer.splatMesh?.scenes?.length ?? 0;
  for (let i = nScenes - 1; i >= 0; i--) {
    try {
      await viewer.removeSplatScene(i, /* update = */ false);
    } catch (_) {
      // older builds may not have removeSplatScene — ignore
    }
  }
}

async function loadSceneFromBuffer(buf) {
  const blob    = new Blob([buf], { type: "application/octet-stream" });
  const blobUrl = URL.createObjectURL(blob);
  try {
    await viewer.addSplatScene(blobUrl, {
      splatAlphaRemovalThreshold: 5,
      showLoadingUI:              false,
      position:                   [0, 0, 0],
      rotation:                   [0, 0, 0, 1],
      scale:                      [1, 1, 1],
    });
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

// ── Display a frame ──────────────────────────────────────────────────────────

async function displayFrame(frameIdx) {
  if (frameIdx === currentDisplayedFrame || isDisplaying) return;
  isDisplaying = true;

  try {
    if (!frameBuffers.has(frameIdx)) {
      await fetchFrame(frameIdx);
    }

    await clearAllScenes();
    await loadSceneFromBuffer(frameBuffers.get(frameIdx));

    currentDisplayedFrame = frameIdx;
    preloadNeighbours(frameIdx);
  } catch (err) {
    console.error(`displayFrame(${frameIdx}) failed:`, err);
  } finally {
    isDisplaying = false;
  }
}

function preloadNeighbours(center) {
  for (let d = 1; d <= PRELOAD_RADIUS; d++) {
    [center + d, center - d].forEach((f) => {
      if (f >= 0 && f < manifest.n_frames && !frameBuffers.has(f)) {
        fetchFrame(f).catch(() => {});
      }
    });
  }
}

// ── File picker ───────────────────────────────────────────────────────────────

async function openSceneFromPicker() {
  const input  = document.createElement("input");
  input.type   = "file";
  input.accept = ".json";
  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    if (!file) return;
    const m = JSON.parse(await file.text());
    await initScene(m);
  });
  input.click();
}

// ── Scene initialisation ──────────────────────────────────────────────────────

async function initScene(m) {
  manifest = m;
  frameBuffers.clear();
  currentDisplayedFrame = -1;

  setLoading(true, "Creating viewer…", 0);
  if (!viewer) createViewer();

  // Prefetch all frames for small scenes, otherwise just the first one.
  const toPreload = m.n_frames <= 20
    ? Array.from({ length: m.n_frames }, (_, i) => i)
    : [0];

  for (let i = 0; i < toPreload.length; i++) {
    setLoading(true, `Fetching frame ${toPreload[i] + 1} / ${m.n_frames}…`,
               (i + 1) / toPreload.length);
    await fetchFrame(toPreload[i]);
  }

  // Display frame 0
  setLoading(true, "Loading scene…", 1);
  await displayFrame(0);

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
  requestAnimationFrame(renderLoop);
}

// ── Render loop ───────────────────────────────────────────────────────────────

function renderLoop(now) {
  requestAnimationFrame(renderLoop);
  timeline?.tick(performance.now() / 1000 - (renderLoop._last ?? 0));
  renderLoop._last = performance.now() / 1000;
  viewer?.update();
  viewer?.render();
  controls?.tickFps();
}

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  document.getElementById("load-btn")
    ?.addEventListener("click", openSceneFromPicker);

  try {
    setLoading(true, "Looking for manifest.json…", 0);
    const m = await loadManifest(MANIFEST_URL);
    await initScene(m);
  } catch (_) {
    setLoading(false);
    console.info("No manifest.json found. Click 'Load Scene' to open one.");
  }
}

boot();
