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

_scene_dir   = None
_frame_paths = None   # list of all frame paths
_saved       = threading.Event()

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SAM2 Seed Picker</title>
<style>
  body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif;
         display: flex; flex-direction: column; align-items: center; padding: 20px; }}
  h2   {{ margin-bottom: 4px; }}
  p    {{ margin: 4px 0 10px; font-size: 13px; color: #aaa; }}
  #scrubber {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; width: 95vw; max-width: 960px; }}
  #scrubber input {{ flex: 1; }}
  #frameLabel {{ font-size: 13px; color: #facc15; min-width: 100px; }}
  #wrap {{ position: relative; cursor: crosshair; display: inline-block; }}
  #frameImg {{ max-width: 95vw; display: block; }}
  canvas {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
  #controls {{ margin-top: 12px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: center; }}
  button {{ padding: 9px 20px; font-size: 14px; border: none; border-radius: 6px; cursor: pointer; }}
  #saveBtn  {{ background: #22c55e; color: #fff; }}
  #clearBtn {{ background: #6b7280; color: #fff; }}
  #modeBtn  {{ background: #3b82f6; color: #fff; min-width: 170px; }}
  #status {{ font-size: 13px; color: #4ade80; min-width: 220px; text-align: center; }}
</style>
</head>
<body>
<h2>SAM2 Seed Picker</h2>
<p>
  Scrub to find a frame where <b>you are visible</b>. Then
  <b>left-click</b> your body (green = foreground).
  Right-click to mark background. Hit <b>Save &amp; Continue</b> when done.
</p>
<div id="scrubber">
  <span style="font-size:13px">Frame:</span>
  <input type="range" id="slider" min="0" max="{n_frames_minus_1}" value="0" oninput="changeFrame(this.value)">
  <span id="frameLabel">1 / {n_frames}</span>
</div>
<div id="wrap">
  <img id="frameImg" src="/frame/0" draggable="false">
  <canvas id="canvas"></canvas>
</div>
<div id="controls">
  <button id="modeBtn" onclick="toggleMode()">Mode: FG (left-click)</button>
  <button id="clearBtn" onclick="clearPoints()">Clear Points</button>
  <button id="saveBtn" onclick="save()">Save &amp; Continue ▶</button>
  <span id="status"></span>
</div>
<script>
const img    = document.getElementById('frameImg');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
const status = document.getElementById('status');
const nFrames = {n_frames};
let points     = [];
let frameIdx   = 0;
let fgOnLeft   = true;

function toggleMode() {{
  fgOnLeft = !fgOnLeft;
  document.getElementById('modeBtn').textContent =
    fgOnLeft ? 'Mode: FG (left-click)' : 'Mode: BG (left-click)';
}}

function changeFrame(v) {{
  frameIdx = parseInt(v);
  document.getElementById('frameLabel').textContent = (frameIdx + 1) + ' / ' + nFrames;
  img.src = '/frame/' + frameIdx + '?t=' + Date.now();
  clearPoints();
}}

img.onload = () => {{
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width  = img.offsetWidth  + 'px';
  canvas.style.height = img.offsetHeight + 'px';
  redraw();
}};

window.addEventListener('resize', () => {{
  canvas.style.width  = img.offsetWidth  + 'px';
  canvas.style.height = img.offsetHeight + 'px';
}});

document.getElementById('wrap').addEventListener('click', (e) => {{
  addPoint(e, fgOnLeft ? 1 : 0);
}});
document.getElementById('wrap').addEventListener('contextmenu', (e) => {{
  e.preventDefault();
  addPoint(e, fgOnLeft ? 0 : 1);
}});

function addPoint(e, label) {{
  const rect = img.getBoundingClientRect();
  const scaleX = img.naturalWidth  / img.offsetWidth;
  const scaleY = img.naturalHeight / img.offsetHeight;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top)  * scaleY);
  points.push({{ x, y, label }});
  redraw();
  const nFg = points.filter(p => p.label===1).length;
  status.textContent = nFg + ' FG point(s) placed';
}}

function redraw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const p of points) {{
    ctx.beginPath();
    ctx.arc(p.x, p.y, 9, 0, Math.PI * 2);
    ctx.fillStyle   = p.label === 1 ? 'rgba(34,197,94,0.85)' : 'rgba(239,68,68,0.85)';
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 2.5;
    ctx.fill();
    ctx.stroke();
  }}
}}

function clearPoints() {{
  points = [];
  redraw();
  status.textContent = '';
}}

async function save() {{
  if (points.filter(p => p.label===1).length === 0) {{
    alert('Please place at least one foreground (green) point on yourself first.');
    return;
  }}
  status.textContent = 'Saving…';
  const res = await fetch('/save', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ frame_index: frameIdx, points }}),
  }});
  const msg = await res.text();
  status.textContent = msg;
  document.getElementById('saveBtn').disabled = true;
  document.getElementById('saveBtn').textContent = '✓ Saved!';
}}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # silence request logs

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            n = len(_frame_paths)
            html = HTML_TEMPLATE.format(n_frames=n, n_frames_minus_1=n - 1)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        elif path.startswith("/frame/"):
            try:
                idx = int(path.split("/frame/")[1])
                fp  = _frame_paths[idx]
            except (ValueError, IndexError):
                self.send_response(404); self.end_headers(); return
            data = fp.read_bytes()
            mime = "image/jpeg" if fp.suffix in (".jpg", ".jpeg") else "image/png"
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

            idx  = body.get("frame_index", 0)
            spec = {
                "frame_index": idx,
                "frame_name":  _frame_paths[idx].name,
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
    global _scene_dir, _frame_paths

    parser = argparse.ArgumentParser(description="Browser-based SAM2 seed picker")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    _scene_dir   = Path(args.scene_dir).resolve()
    frames_dir   = _scene_dir / "frames"
    _frame_paths = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))
    if not _frame_paths:
        sys.exit(f"No frames found in {frames_dir}")

    print(f"[seed_ui] {len(_frame_paths)} frames found in {frames_dir}")
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
