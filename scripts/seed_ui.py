"""
seed_ui.py — Browser-based SAM2 seed point picker

Opens http://localhost:7860 with frame 0 of the scene.
Click to add foreground points (green), right-click for background (red).
Press "Save & Continue" to write mask_frame0.json and exit.

Usage:
    python scripts/seed_ui.py --scene-dir ./my_scene/
"""

import argparse
import base64
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ─── Find frame 0 ────────────────────────────────────────────────────────────

def find_frame0(scene_dir: Path):
    frames = sorted((scene_dir / "frames").glob("*.png"))
    if not frames:
        frames = sorted((scene_dir / "frames").glob("*.jpg"))
    if not frames:
        sys.exit(f"No frames found in {scene_dir / 'frames'}")
    return frames[0]


# ─── Tiny HTTP server ─────────────────────────────────────────────────────────

_scene_dir  = None
_frame_path = None
_saved      = threading.Event()

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SAM2 Seed Picker</title>
<style>
  body { margin: 0; background: #111; color: #eee; font-family: sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 20px; }
  h2   { margin-bottom: 6px; }
  p    { margin: 4px 0 12px; font-size: 13px; color: #aaa; }
  #wrap { position: relative; cursor: crosshair; display: inline-block; }
  #frame { max-width: 95vw; display: block; }
  canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
  #controls { margin-top: 14px; display: flex; gap: 12px; align-items: center; }
  button { padding: 10px 22px; font-size: 15px; border: none; border-radius: 6px;
           cursor: pointer; }
  #saveBtn  { background: #22c55e; color: #fff; }
  #clearBtn { background: #6b7280; color: #fff; }
  #modeBtn  { background: #3b82f6; color: #fff; min-width: 160px; }
  #status { font-size: 13px; color: #4ade80; min-width: 200px; text-align: center; }
</style>
</head>
<body>
<h2>SAM2 Seed Picker — Frame 0</h2>
<p>
  <b>Left-click</b> = foreground (green dot) &nbsp;|&nbsp;
  <b>Right-click</b> = background (red dot) &nbsp;|&nbsp;
  Mode button toggles which click adds which label.
</p>
<div id="wrap">
  <img id="frame" src="/frame" draggable="false">
  <canvas id="canvas"></canvas>
</div>
<div id="controls">
  <button id="modeBtn" onclick="toggleMode()">Mode: FG (left-click)</button>
  <button id="clearBtn" onclick="clearPoints()">Clear All</button>
  <button id="saveBtn" onclick="save()">Save &amp; Continue ▶</button>
  <span id="status"></span>
</div>
<script>
const img    = document.getElementById('frame');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
const status = document.getElementById('status');
let points   = [];
let fgOnLeft = true;   // if false, left=bg, right=fg

function toggleMode() {
  fgOnLeft = !fgOnLeft;
  document.getElementById('modeBtn').textContent =
    fgOnLeft ? 'Mode: FG (left-click)' : 'Mode: BG (left-click)';
}

img.onload = () => {
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width  = img.offsetWidth  + 'px';
  canvas.style.height = img.offsetHeight + 'px';
  redraw();
};

window.addEventListener('resize', () => {
  canvas.style.width  = img.offsetWidth  + 'px';
  canvas.style.height = img.offsetHeight + 'px';
});

document.getElementById('wrap').addEventListener('click', (e) => {
  addPoint(e, fgOnLeft ? 1 : 0);
});
document.getElementById('wrap').addEventListener('contextmenu', (e) => {
  e.preventDefault();
  addPoint(e, fgOnLeft ? 0 : 1);
});

function addPoint(e, label) {
  const rect = img.getBoundingClientRect();
  const scaleX = img.naturalWidth  / img.offsetWidth;
  const scaleY = img.naturalHeight / img.offsetHeight;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top)  * scaleY);
  points.push({ x, y, label });
  redraw();
  status.textContent = `${points.length} point(s) placed`;
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const p of points) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
    ctx.fillStyle   = p.label === 1 ? 'rgba(34,197,94,0.85)' : 'rgba(239,68,68,0.85)';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 2;
    ctx.fill();
    ctx.stroke();
  }
}

function clearPoints() {
  points = [];
  redraw();
  status.textContent = 'Cleared';
}

async function save() {
  if (points.length === 0) {
    alert('Please click at least one foreground point first.');
    return;
  }
  status.textContent = 'Saving…';
  const res = await fetch('/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ points }),
  });
  const msg = await res.text();
  status.textContent = msg;
  document.getElementById('saveBtn').disabled = true;
  document.getElementById('saveBtn').textContent = '✓ Saved!';
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # silence request logs

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif self.path == "/frame":
            data = _frame_path.read_bytes()
            ext  = _frame_path.suffix.lstrip(".")
            mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/save":
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))

            spec = {
                "frame_index": 0,
                "frame_name":  _frame_path.name,
                "points":      body["points"],
            }
            out = _scene_dir / "mask_frame0.json"
            out.write_text(json.dumps(spec, indent=2))

            n_fg = sum(1 for p in body["points"] if p["label"] == 1)
            n_bg = sum(1 for p in body["points"] if p["label"] == 0)
            msg  = f"Saved {n_fg} FG + {n_bg} BG points → {out}"
            print(f"\n[seed_ui] {msg}")

            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg.encode())

            _saved.set()   # signal main thread to exit
        else:
            self.send_response(404); self.end_headers()


def main():
    global _scene_dir, _frame_path

    parser = argparse.ArgumentParser(description="Browser-based SAM2 seed picker")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _scene_dir  = Path(args.scene_dir).resolve()
    _frame_path = find_frame0(_scene_dir)

    print(f"[seed_ui] Frame 0: {_frame_path.name}")
    print(f"[seed_ui] Opening http://localhost:{args.port}")
    print(f"[seed_ui] Left-click = foreground (green), Right-click = background (red)")
    print(f"[seed_ui] Click 'Save & Continue' when done.\n")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=lambda: webbrowser.open(f"http://localhost:{args.port}"),
                     daemon=True).start()

    # Serve until the user saves
    while not _saved.is_set():
        server.handle_request()

    server.server_close()
    print("[seed_ui] Done. You can now run Stage 2:")
    print(f"  python3 pipeline/02_segment.py --scene-dir {_scene_dir}")


if __name__ == "__main__":
    main()
