/**
 * main.js — 4D Scene Viewer
 *
 * Frame switching strategy: add new scene FIRST (old stays visible),
 * then remove old scenes — eliminates the black-flash between frames.
 * Background prefetch keeps next frames cached for instant switching.
 */

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import { Timeline } from "./timeline.js";
import { Controls } from "./controls.js";

const MANIFEST_URL = "./manifest.json";

// ── State ─────────────────────────────────────────────────────────────────────

let viewer   = null;
let manifest = null;
let timeline = null;
let controls = null;
let viewerStarted = false;

let currentDisplayedFrame = -1;
let isDisplaying = false;

// ArrayBuffer cache: frameIndex → ArrayBuffer
const frameCache = new Map();

// ── Loading overlay ───────────────────────────────────────────────────────────

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

// ── Manifest ──────────────────────────────────────────────────────────────────

async function loadManifest(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`manifest fetch failed: ${res.status}`);
  return res.json();
}

// ── Viewer ────────────────────────────────────────────────────────────────────

function createViewer() {
  const container = document.getElementById("canvas-container");
  viewer = new GaussianSplats3D.Viewer({
    rootElement:            container,
    selfDrivenMode:         true,
    useBuiltInControls:     true,
    initialCameraPosition:  [-0.2, 0.0, 0.0],
    initialCameraLookAt:    [-0.02, 0.01, 0.245],
    cameraUp:               [0, -1, 0],
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort:     false,
    logLevel:               GaussianSplats3D.LogLevel.Warning,
  });
}

// ── Splat loading ─────────────────────────────────────────────────────────────

const SPLAT_OPTS = {
  format:                     GaussianSplats3D.SceneFormat.Splat,
  splatAlphaRemovalThreshold: 5,
  showLoadingUI:              false,
  position:                   [0, 0, 0],
  rotation:                   [0, 0, 0, 1],
  scale:                      [1, 1, 1],
};

async function addSplatFromUrl(url) {
  await viewer.addSplatScene(url, SPLAT_OPTS);
}

async function addSplatFromBuffer(buf) {
  const blob = new Blob([buf], { type: "application/octet-stream" });
  const url  = URL.createObjectURL(blob);
  try   { await viewer.addSplatScene(url, SPLAT_OPTS); }
  finally { URL.revokeObjectURL(url); }
}

// ── Prefetch ──────────────────────────────────────────────────────────────────

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

// ── Display a frame ───────────────────────────────────────────────────────────

async function displayFrame(frameIdx) {
  if (frameIdx === currentDisplayedFrame || isDisplaying) return true;
  isDisplaying = true;
  try {
    // Add NEW scene first — old scene stays visible, no black flash
    if (frameCache.has(frameIdx)) {
      await addSplatFromBuffer(frameCache.get(frameIdx));
    } else {
      await addSplatFromUrl(`./${manifest.files[frameIdx].file}`);
    }

    // Remove all OLD scenes now that new one is showing
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

// ── Render loop (timeline + FPS — viewer drives its own render) ───────────────

function renderLoop() {
  requestAnimationFrame(renderLoop);
  timeline?.tick(performance.now() / 1000 - (renderLoop._t ?? 0));
  renderLoop._t = performance.now() / 1000;
  controls?.tickFps();
}

// ── File picker ───────────────────────────────────────────────────────────────

async function openSceneFromPicker() {
  const input = document.createElement("input");
  input.type  = "file";
  input.accept = ".json";
  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    if (!file) return;
    try { await initScene(JSON.parse(await file.text())); }
    catch (e) { showError(e.message); }
  });
  input.click();
}

// ── Scene init ────────────────────────────────────────────────────────────────

async function initScene(m) {
  manifest = m;
  currentDisplayedFrame = -1;
  viewerStarted = false;
  frameCache.clear();

  if (!viewer) createViewer();

  setLoading(true, `Loading frame 1 of ${m.n_frames}…`, 0);
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
}

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  document.getElementById("load-btn")
    ?.addEventListener("click", openSceneFromPicker);
  try {
    setLoading(true, "Looking for manifest.json…", 0);
    const m = await loadManifest(MANIFEST_URL);
    await initScene(m);
  } catch (err) {
    showError(`Could not load scene: ${err.message}`);
  }
}

boot();
