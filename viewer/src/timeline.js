/**
 * timeline.js — Timeline scrubber and playback controller
 *
 * Manages:
 *   - play / pause state
 *   - speed multiplier (0.25×, 0.5×, 1×, 2×)
 *   - loop toggle
 *   - frame stepping (+1 / -1)
 *   - emits "seek" events when the user scrubs or playback advances
 */

export class Timeline {
  /**
   * @param {object} opts
   * @param {number}   opts.nFrames   - total number of frames
   * @param {number[]} opts.timestamps - array of timestamp values (seconds), length nFrames
   * @param {Function} opts.onSeek     - called with (frameIndex: number) when frame changes
   */
  constructor({ nFrames, timestamps, onSeek }) {
    this.nFrames    = nFrames;
    this.timestamps = timestamps;
    this.onSeek     = onSeek;

    this.currentFrame = 0;
    this.isPlaying    = false;
    this.speed        = 1.0;
    this.looping      = true;

    // Playback accumulator: fractional frame position
    this._frameAccum    = 0;
    this._lastTickTime  = null;

    // Assume 30 fps display → base advance rate 1 frame per ~33ms, scaled by speed
    this._baseFramesPerSecond = 15;   // advance at 15 scene-fps when speed=1×

    this._bindElements();
    this._bindKeys();
    this.updateUI();
  }

  // ── Internal: bind DOM elements ────────────────────────────────────────────

  _bindElements() {
    this.slider    = document.getElementById("timeline-slider");
    this.playBtn   = document.getElementById("play-btn");
    this.loopBtn   = document.getElementById("loop-btn");
    this.speedSel  = document.getElementById("speed-select");
    this.resetBtn  = document.getElementById("reset-btn");
    this.tsDisplay = document.getElementById("timestamp-display");

    this.slider.max   = String(this.nFrames - 1);
    this.slider.value = "0";

    this.slider.addEventListener("input", () => {
      const f = parseInt(this.slider.value, 10);
      this._frameAccum = f;
      this._seek(f, /* fromSlider */ true);
    });

    this.playBtn.addEventListener("click", () => this.togglePlay());
    this.loopBtn.addEventListener("click", () => this.toggleLoop());
    this.speedSel.addEventListener("change", () => {
      this.speed = parseFloat(this.speedSel.value);
    });
    this.resetBtn.addEventListener("click", () => this.seekToFrame(0));
  }

  _bindKeys() {
    window.addEventListener("keydown", (e) => {
      // Ignore when typing in an input
      if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

      switch (e.key) {
        case " ":
          e.preventDefault();
          this.togglePlay();
          break;
        case "ArrowRight":
          e.preventDefault();
          this.stepFrame(+1);
          break;
        case "ArrowLeft":
          e.preventDefault();
          this.stepFrame(-1);
          break;
        case "ArrowUp":
          e.preventDefault();
          this.stepFrame(+5);
          break;
        case "ArrowDown":
          e.preventDefault();
          this.stepFrame(-5);
          break;
        case "Home":
          e.preventDefault();
          this.seekToFrame(0);
          break;
        case "End":
          e.preventDefault();
          this.seekToFrame(this.nFrames - 1);
          break;
      }
    });
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  togglePlay() {
    this.isPlaying = !this.isPlaying;
    if (this.isPlaying) {
      this._lastTickTime = null;  // reset so first tick doesn't jump
    }
    this.updateUI();
  }

  toggleLoop() {
    this.looping = !this.looping;
    this.updateUI();
  }

  seekToFrame(f) {
    f = Math.max(0, Math.min(this.nFrames - 1, Math.round(f)));
    this._frameAccum = f;
    this._seek(f);
  }

  stepFrame(delta) {
    const was = this.isPlaying;
    this.isPlaying = false;
    this.seekToFrame(this.currentFrame + delta);
    this.isPlaying = was;
    this.updateUI();
  }

  /**
   * Advance playback by dt seconds. Call this once per animation frame.
   * @param {number} dt  - elapsed seconds since last call
   */
  tick(dt) {
    if (!this.isPlaying) return;

    const advance = dt * this._baseFramesPerSecond * this.speed;
    this._frameAccum += advance;

    if (this._frameAccum >= this.nFrames) {
      if (this.looping) {
        this._frameAccum = this._frameAccum % this.nFrames;
      } else {
        this._frameAccum = this.nFrames - 1;
        this.isPlaying   = false;
        this.updateUI();
      }
    }

    const newFrame = Math.floor(this._frameAccum);
    if (newFrame !== this.currentFrame) {
      this._seek(newFrame);
    }
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  _seek(frameIdx, fromSlider = false) {
    this.currentFrame = frameIdx;

    // Update slider
    if (!fromSlider) {
      this.slider.value = String(frameIdx);
    }

    // Update timestamp display
    const ts = this.timestamps?.[frameIdx] ?? frameIdx;
    this.tsDisplay.textContent = `t = ${ts.toFixed(3)}s`;

    // Notify main
    this.onSeek(frameIdx);
  }

  updateUI() {
    this.playBtn.textContent = this.isPlaying ? "⏸" : "▶";
    if (this.looping) {
      this.loopBtn.classList.add("active");
    } else {
      this.loopBtn.classList.remove("active");
    }
  }
}
