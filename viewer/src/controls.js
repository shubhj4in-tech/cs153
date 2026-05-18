/**
 * controls.js — Orbit controls, reset view, and screenshot
 *
 * Three.js OrbitControls are already provided by GaussianSplats3D's viewer.
 * This module handles:
 *   - Storing / restoring the initial camera pose (reset view)
 *   - Screenshot capture (canvas → PNG download)
 *   - Binding the UI buttons
 */

import * as THREE from "three";

export class Controls {
  /**
   * @param {object} opts
   * @param {THREE.Camera}   opts.camera
   * @param {THREE.Renderer} opts.renderer
   * @param {THREE.OrbitControls|null} opts.orbitControls - may be null if GS3D manages them
   */
  constructor({ camera, renderer, orbitControls = null }) {
    this.camera        = camera;
    this.renderer      = renderer;
    this.orbitControls = orbitControls;

    // Snapshot initial pose after first frame so we have something to reset to
    this._initialPos    = null;
    this._initialTarget = null;

    this._bindButtons();
    this._initFpsCounter();
  }

  // Capture the current camera position as the "home" pose (call once scene is loaded)
  saveHomeCamera() {
    this._initialPos    = this.camera.position.clone();
    this._initialTarget = this.orbitControls?.target?.clone()
                         ?? new THREE.Vector3(0, 0, 0);
  }

  resetCamera() {
    if (!this._initialPos) return;
    this.camera.position.copy(this._initialPos);
    if (this.orbitControls) {
      this.orbitControls.target.copy(this._initialTarget);
      this.orbitControls.update();
    }
  }

  /**
   * Grab the current renderer canvas and download it as a PNG.
   */
  screenshot() {
    const canvas = this.renderer.domElement;
    const link   = document.createElement("a");
    link.download = `4drecon_frame_${Date.now()}.png`;
    link.href     = canvas.toDataURL("image/png");
    link.click();

    // Flash effect
    const flash = document.getElementById("flash");
    if (flash) {
      flash.classList.add("flash-anim");
      setTimeout(() => flash.classList.remove("flash-anim"), 200);
    }
  }

  // ── Internal ───────────────────────────────────────────────────────────────

  _bindButtons() {
    const resetBtn = document.getElementById("reset-btn");
    if (resetBtn) {
      resetBtn.addEventListener("click", () => this.resetCamera());
    }

    const ssBtn = document.getElementById("screenshot-btn");
    if (ssBtn) {
      ssBtn.addEventListener("click", () => this.screenshot());
    }
  }

  _initFpsCounter() {
    this._fpsEl     = document.getElementById("fps-counter");
    this._frameCount = 0;
    this._fpsLastTime = performance.now();
  }

  /**
   * Call once per animation frame to update the FPS display.
   */
  tickFps() {
    this._frameCount++;
    const now = performance.now();
    const elapsed = now - this._fpsLastTime;
    if (elapsed >= 500) {
      const fps = Math.round(this._frameCount / (elapsed / 1000));
      if (this._fpsEl) this._fpsEl.textContent = `${fps} fps`;
      this._frameCount  = 0;
      this._fpsLastTime = now;
    }
  }
}
