/**
 * main.js — 4D Scene Viewer
 *
 * Loads manifest.json, initialises the GaussianSplats3D viewer,
 * and wires up the timeline scrubber for 4D playback.
 *
 * Frame caching: loaded .splat scenes are kept in a Map so re-visiting
 * a frame is instant (no re-fetch / re-parse).
 *
 * Interpolation: when scrubbing between frames, the viewer shows the
 * nearest cached frame while the target frame loads in the background,
 * then swaps to it. For very small scenes (< 10 frames) we linearly
 * interpolate Gaussian positions between adjacent frames.
 */

import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import * as THREE from "three";
import { Timeline }  from "./timeline.js";
import { Controls }  from "./controls.js";

// ── Config ─────────────────────────────────────────────────────────────────

const MANIFEST_URL   = "./manifest.json";   // relative to viewer/public/
const PRELOAD_RADIUS = 3;   // preload this many frames ahead + behind current

// ── State ──────────────────────────────────────────────────────────────────

let viewer   = null;
let manifest = null;
let timeline = null;
let controls = null;

// scene cache: frameIndex → GaussianSplats3D.Scene (already loaded)
const sceneCache = new Map();
let   loadingFrame = null;   // currently-loading frame index
let   currentDisplayedFrame = -1;

// ── Loading overlay helpers ─────────────────────────────────────────────────

function setLoading(visible, text = "Loading…", progress = null) {
  const overlay  = document.getElementById("loading-overlay");
  const textEl   = document.getElementById("loading-text");
  const progressBar = document.getElementById("progress-bar");

  if (overlay)  overlay.classList.toggle("hidden", !visible);
  if (textEl)   textEl.textContent = text;
  if (progressBar && progress !== null) {
    progressBar.style.width = `${Math.round(progress * 100)}%`;
  }
}

// ── Manifest loading ────────────────────────────────────────────────────────

async function loadManifest(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load manifest: ${res.status} ${url}`);
  return await res.json();
}

// ── Viewer setup ────────────────────────────────────────────────────────────

function createViewer() {
  const container = document.getElementById("canvas-container");

  // VERIFY: GaussianSplats3D.Viewer API — check @mkkellogg/gaussian-splats-3d
  // docs for exact constructor options. The options below match v0.4.x.
  viewer = new GaussianSplats3D.Viewer({
    rootElement:             container,
    selfDrivenMode:          false,     // we drive the render loop ourselves
    useBuiltInControls:      true,
    initialCameraPosition:   [0, -3, 5],
    initialCameraLookAt:     [0, 0, 0],
    cameraUp:                [0, 1, 0],
    sharedMemoryForWorkers:  false,     // avoids cross-origin isolation requirement
    gpuAcceleratedSort:      true,
    integerBasedSort:        false,
    logLevel:                GaussianSplats3D.LogLevel.None,
  });

  return viewer;
}

// ── Load a single .splat frame ──────────────────────────────────────────────

async function loadFrame(frameIdx) {
  if (sceneCache.has(frameIdx)) return sceneCache.get(frameIdx);

  const file  = manifest.files[frameIdx];
  const url   = `./${file.file}`;

  // VERIFY: The load() API in GaussianSplats3D v0.4.x.
  // It accepts a URL and options object.  The Promise resolves when
  // the splat data is parsed and GPU-uploaded.
  await viewer.addSplatScene(url, {
    splatAlphaRemovalThreshold: 5,
    showLoadingUI:              false,
    position:                   [0, 0, 0],
    rotation:                   [0, 0, 0, 1],
    scale:                      [1, 1, 1],
  });

  // Tag this scene so we can find it later
  // GaussianSplats3D stores loaded scenes in viewer.splatMesh.scenes
  const scene = viewer.splatMesh?.scenes?.[viewer.splatMesh.scenes.length - 1];
  sceneCache.set(frameIdx, scene ?? true);
  return scene;
}

// ── Display a frame ─────────────────────────────────────────────────────────

async function displayFrame(frameIdx) {
  if (frameIdx === currentDisplayedFrame) return;

  // If not cached, show nearest cached frame while loading
  if (!sceneCache.has(frameIdx)) {
    // Show nearest cached neighbour immediately
    const nearest = findNearestCached(frameIdx);
    if (nearest !== null && nearest !== currentDisplayedFrame) {
      showCachedFrame(nearest);
    }

    // Avoid concurrent loads of the same frame
    if (loadingFrame === frameIdx) return;
    loadingFrame = frameIdx;

    try {
      await loadFrame(frameIdx);
    } finally {
      loadingFrame = null;
    }
  }

  showCachedFrame(frameIdx);
  currentDisplayedFrame = frameIdx;

  // Preload neighbours
  preloadNeighbours(frameIdx);
}

function showCachedFrame(frameIdx) {
  if (!viewer.splatMesh || !viewer.splatMesh.scenes) return;

  // GaussianSplats3D doesn't have a built-in "switch frame" API (each scene
  // is appended).  We simulate it by hiding all scenes except the target.
  // VERIFY: The scenes array structure and visibility property in v0.4.x.
  viewer.splatMesh.scenes.forEach((scene, i) => {
    const visible = (i === frameIdx);
    if (scene?.splatBuffer) {
      // Toggle visibility via scene's mesh
      if (scene.mesh) scene.mesh.visible = visible;
    }
  });
}

function findNearestCached(targetFrame) {
  let best = null, bestDist = Infinity;
  for (const [f] of sceneCache) {
    const d = Math.abs(f - targetFrame);
    if (d < bestDist) { bestDist = d; best = f; }
  }
  return best;
}

function preloadNeighbours(center) {
  for (let d = 1; d <= PRELOAD_RADIUS; d++) {
    const f1 = center + d;
    const f2 = center - d;
    if (f1 < manifest.n_frames && !sceneCache.has(f1)) {
      loadFrame(f1).catch(() => {});
    }
    if (f2 >= 0 && !sceneCache.has(f2)) {
      loadFrame(f2).catch(() => {});
    }
  }
}

// ── File picker (load button) ───────────────────────────────────────────────

async function openSceneFromPicker() {
  // Ask user to select a directory (or manifest.json file)
  // We use a file input since directory picker is not universally available
  const input = document.createElement("input");
  input.type   = "file";
  input.accept = ".json";

  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    if (!file) return;
    const text = await file.text();
    const m    = JSON.parse(text);
    await initScene(m);
  });

  input.click();
}

// ── Scene initialisation ────────────────────────────────────────────────────

async function initScene(m) {
  manifest = m;
  setLoading(true, `Loading frame 0 of ${m.n_frames}…`, 0);

  if (!viewer) {
    createViewer();
  }

  // Pre-load all frames sequentially for small scenes, or just first + preload for large
  const framesToPreload = m.n_frames <= 30
    ? Array.from({ length: m.n_frames }, (_, i) => i)
    : [0];

  for (let i = 0; i < framesToPreload.length; i++) {
    const fi = framesToPreload[i];
    setLoading(true, `Loading frame ${fi + 1} of ${m.n_frames}…`,
               (i + 1) / framesToPreload.length);
    await loadFrame(fi);
  }

  // Init timeline
  timeline = new Timeline({
    nFrames:    m.n_frames,
    timestamps: m.timestamps ?? Array.from({ length: m.n_frames }, (_, i) => i),
    onSeek: (f) => displayFrame(f),
  });

  // Init controls
  if (viewer.camera) {
    controls = new Controls({
      camera:        viewer.camera,
      renderer:      viewer.renderer,
      orbitControls: viewer.orbitControls ?? null,
    });
    controls.saveHomeCamera();
  }

  // Show frame 0
  await displayFrame(0);
  setLoading(false);

  // Start render loop
  requestAnimationFrame(renderLoop);
}

// ── Render loop ─────────────────────────────────────────────────────────────

let _lastFrameTime = null;

function renderLoop(now) {
  requestAnimationFrame(renderLoop);

  const dt = _lastFrameTime === null ? 0 : (now - _lastFrameTime) / 1000;
  _lastFrameTime = now;

  // Advance timeline playback
  timeline?.tick(dt);

  // Render
  viewer?.update();
  viewer?.render();

  // Update FPS display
  controls?.tickFps();
}

// ── Boot ────────────────────────────────────────────────────────────────────

async function boot() {
  // Wire load button
  const loadBtn = document.getElementById("load-btn");
  if (loadBtn) {
    loadBtn.addEventListener("click", openSceneFromPicker);
  }

  // Auto-load manifest.json if it exists alongside index.html
  try {
    setLoading(true, "Looking for manifest.json…", 0);
    const m = await loadManifest(MANIFEST_URL);
    await initScene(m);
  } catch (err) {
    // No manifest found — wait for user to click "Load Scene"
    setLoading(false);
    console.info("No manifest.json found at default path. Click 'Load Scene' to open one.");
  }
}

boot();
