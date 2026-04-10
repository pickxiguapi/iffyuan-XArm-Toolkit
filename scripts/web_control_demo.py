#!/usr/bin/env python3
"""XArm 键盘控制 Web Demo — 双相机实时画面 + 键盘遥操作.

启动 (真机):  python scripts/web_control_demo.py
启动 (调试):  python scripts/web_control_demo.py --mock
打开:         http://localhost:8080

键盘映射 (网页内按键):
  W/S — Y 前/后      A/D — X 左/右      Q/E — Z 上/下
  I/K — Roll ±       J/L — Pitch ±      U/O — Yaw ±
  Space — 切换夹爪    R — 复位到 Home     Esc — 急停

--mock 模式下不连接真机 / 相机，用于本地 UI 调试。
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Mock env for local testing (Machine A, no hardware)
# ---------------------------------------------------------------------------

class MockXArmEnv:
    """Fake env that mimics XArmEnv interface for UI development."""

    RESET_POSE = np.array([470.0, 0.0, 530.0, math.pi, 0.0, -math.pi / 2])

    def __init__(self, **_kw):
        self.goal_pos = self.RESET_POSE.copy()
        self._gripper = 840.0
        self._servo = np.zeros(6)
        print("[MockXArmEnv] 初始化完成 (模拟模式, 无真机连接)")

    def reset(self, close_gripper=True):
        self.goal_pos = self.RESET_POSE.copy()
        self._gripper = 0.0 if close_gripper else 840.0
        return self._obs()

    def step(self, action, gripper_action=None, speed=1000, joint_speed=1.0):
        self.goal_pos += np.asarray(action, dtype=np.float64)
        if gripper_action is not None:
            self._gripper = float(gripper_action)
        return self._obs()

    def _obs(self):
        return {
            "cart_pos": self.goal_pos.copy(),
            "servo_angle": self._servo.copy(),
            "ext_force": None,
            "raw_force": None,
            "goal_pos": self.goal_pos.copy(),
            "gripper_position": np.array(self._gripper),
        }

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Camera background capture thread
# ---------------------------------------------------------------------------

class CameraThread(threading.Thread):
    """Daemon thread: continuously captures JPEG frames from a RealsenseEnv."""

    def __init__(self, name: str, cam_env):
        super().__init__(daemon=True, name=f"cam-{name}")
        self.cam_name = name
        self.cam_env = cam_env
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                cam_obs = self.cam_env.step()
                rgb = np.asarray(cam_obs["rgb"])  # H×W×3 uint8, RGB order
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with cam_lock:
                        cam_frames[self.cam_name] = jpeg.tobytes()
            except Exception as e:
                print(f"[CameraThread-{self.cam_name}] {e}")
                time.sleep(0.1)

    def stop(self):
        self._stop_event.set()


def _make_placeholder_frame(label: str) -> bytes:
    """Generate a 640×480 dark placeholder JPEG with centered label."""
    img = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.putText(img, label, (160, 230), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (100, 100, 100), 2, cv2.LINE_AA)
    cv2.putText(img, "MOCK MODE", (200, 290), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (80, 80, 80), 1, cv2.LINE_AA)
    _, jpeg = cv2.imencode(".jpg", img)
    return jpeg.tobytes()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

env = None
obs: dict | None = None
lock = threading.Lock()
gripper_open = True  # 夹爪当前状态
STEP_POS = 2.0       # mm per keypress — 非常小，安全试探
STEP_ROT = 0.005     # rad per keypress (~0.3°)

# Camera frames (JPEG bytes), updated by CameraThread
cam_frames: dict[str, bytes | None] = {"arm": None, "fix": None}
cam_lock = threading.Lock()
cam_threads: list[CameraThread] = []

# Action table: key → 6D delta [dx, dy, dz, droll, dpitch, dyaw]
ACTION_MAP = {
    "a": np.array([-STEP_POS, 0, 0, 0, 0, 0]),   # X-
    "d": np.array([STEP_POS, 0, 0, 0, 0, 0]),     # X+
    "w": np.array([0, STEP_POS, 0, 0, 0, 0]),     # Y+
    "s": np.array([0, -STEP_POS, 0, 0, 0, 0]),    # Y-
    "q": np.array([0, 0, STEP_POS, 0, 0, 0]),     # Z+
    "e": np.array([0, 0, -STEP_POS, 0, 0, 0]),    # Z-
    "i": np.array([0, 0, 0, STEP_ROT, 0, 0]),     # Roll+
    "k": np.array([0, 0, 0, -STEP_ROT, 0, 0]),    # Roll-
    "j": np.array([0, 0, 0, 0, STEP_ROT, 0]),     # Pitch+
    "l": np.array([0, 0, 0, 0, -STEP_ROT, 0]),    # Pitch-
    "u": np.array([0, 0, 0, 0, 0, STEP_ROT]),     # Yaw+
    "o": np.array([0, 0, 0, 0, 0, -STEP_ROT]),    # Yaw-
}


def handle_key(key: str) -> dict:
    """Process a single keypress and return updated status."""
    global obs, gripper_open

    with lock:
        if key in ACTION_MAP:
            obs = env.step(action=ACTION_MAP[key])
            return _status("ok", f"move {key}")

        if key == " ":
            gripper_open = not gripper_open
            g = 840 if gripper_open else 0
            obs = env.step(action=[0, 0, 0, 0, 0, 0], gripper_action=g)
            return _status("ok", "夹爪 " + ("打开" if gripper_open else "关闭"))

        if key == "r":
            obs = env.reset(close_gripper=True)
            gripper_open = False
            return _status("ok", "已复位")

        if key == "escape":
            # 急停: 发送零动作
            obs = env.step(action=[0, 0, 0, 0, 0, 0])
            return _status("warn", "急停!")

    return _status("ignore", "")


def _status(level: str, msg: str) -> dict:
    """Build JSON-serializable status dict."""
    cp = obs["cart_pos"] if obs else [0] * 6
    gp = obs["gripper_position"] if obs else 0
    return {
        "level": level,
        "msg": msg,
        "pos": {
            "x": round(float(cp[0]), 1),
            "y": round(float(cp[1]), 1),
            "z": round(float(cp[2]), 1),
            "roll": round(float(math.degrees(cp[3])), 1),
            "pitch": round(float(math.degrees(cp[4])), 1),
            "yaw": round(float(math.degrees(cp[5])), 1),
        },
        "gripper": round(float(np.asarray(gp).item()), 1),
        "gripper_open": gripper_open,
    }


# ---------------------------------------------------------------------------
# HTML — 双相机 + 键盘控制面板
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XArm 键盘控制</title>
<style>
  :root { --bg: #0a0a0a; --card: #161616; --accent: #3b82f6; --green: #22c55e;
          --red: #ef4444; --yellow: #eab308; --text: #e4e4e7; --dim: #71717a; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
         background: var(--bg); color: var(--text); min-height:100vh;
         display:flex; flex-direction:column; align-items:center;
         padding: 24px 16px; user-select:none; }
  h1 { font-size: 18px; font-weight: 600; letter-spacing: 2px;
       color: var(--accent); margin-bottom: 16px; }

  /* Status bar */
  #status { font-size: 13px; height: 20px; margin-bottom: 16px;
            color: var(--dim); transition: color .15s; }
  #status.warn { color: var(--yellow); }

  /* Main two-column layout */
  .main-layout { display: flex; gap: 28px; align-items: flex-start;
                 max-width: 1100px; width: 100%; }

  /* Left: camera feeds */
  .cam-column { display: flex; flex-direction: column; gap: 10px;
                flex-shrink: 0; }
  .cam-box { position: relative; }
  .cam-box img { width: 480px; height: 360px; border-radius: 8px;
                 background: var(--card); object-fit: cover; display: block; }
  .cam-label { position: absolute; top: 8px; left: 10px;
               font-size: 11px; color: #fff; background: rgba(0,0,0,0.55);
               padding: 2px 8px; border-radius: 4px; }

  /* Right: controls panel */
  .ctrl-column { flex: 1; display: flex; flex-direction: column;
                 align-items: center; }

  /* Pose display */
  .pose-grid { display: grid; grid-template-columns: repeat(3, 1fr);
               gap: 8px; margin-bottom: 20px; width: 100%; max-width: 420px; }
  .pose-cell { background: var(--card); border-radius: 8px; padding: 10px 14px;
               text-align: center; }
  .pose-label { font-size: 11px; color: var(--dim); margin-bottom: 2px; }
  .pose-val   { font-size: 20px; font-weight: 700; }

  /* Gripper indicator */
  .gripper-bar { display:flex; align-items:center; gap:10px;
                 margin-bottom: 24px; font-size: 13px; color: var(--dim); }
  .gripper-dot { width: 10px; height: 10px; border-radius: 50%; }
  .gripper-dot.open  { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .gripper-dot.close { background: var(--red); box-shadow: 0 0 6px var(--red); }

  /* Keyboard layout hint */
  .key-hint { display: grid; gap: 6px; margin-bottom: 14px; }
  .key-row  { display: flex; justify-content: center; gap: 6px; }
  .key { width: 42px; height: 42px; border-radius: 6px; display:flex;
         align-items:center; justify-content:center; font-size: 14px;
         font-weight: 600; background: var(--card); border: 1px solid #333;
         transition: all .08s; }
  .key.active { background: var(--accent); border-color: var(--accent);
                color: #fff; transform: scale(0.93); }
  .key.wide { width: 120px; }
  .key-group-label { font-size: 11px; color: var(--dim); text-align: center;
                     margin: 6px 0 4px; }

  /* Footer */
  .footer { margin-top: auto; padding-top: 20px; font-size: 11px;
            color: var(--dim); }
  .mock-badge { display:inline-block; background:#854d0e; color:#fef08a;
                padding:2px 8px; border-radius:4px; font-size:11px; }
</style>
</head>
<body>

<h1>XARM KEYBOARD CTRL</h1>
<div id="mode"></div>
<div id="status">按任意控制键开始 ...</div>

<div class="main-layout">

  <!-- Left: Camera feeds -->
  <div class="cam-column">
    <div class="cam-box">
      <div class="cam-label">ARM (D435i)</div>
      <img src="/stream/arm" alt="arm camera">
    </div>
    <div class="cam-box">
      <div class="cam-label">FIX (L515)</div>
      <img src="/stream/fix" alt="fix camera">
    </div>
  </div>

  <!-- Right: Controls -->
  <div class="ctrl-column">
    <div class="pose-grid">
      <div class="pose-cell"><div class="pose-label">X (mm)</div><div class="pose-val" id="px">—</div></div>
      <div class="pose-cell"><div class="pose-label">Y (mm)</div><div class="pose-val" id="py">—</div></div>
      <div class="pose-cell"><div class="pose-label">Z (mm)</div><div class="pose-val" id="pz">—</div></div>
      <div class="pose-cell"><div class="pose-label">Roll (°)</div><div class="pose-val" id="pr">—</div></div>
      <div class="pose-cell"><div class="pose-label">Pitch (°)</div><div class="pose-val" id="pp">—</div></div>
      <div class="pose-cell"><div class="pose-label">Yaw (°)</div><div class="pose-val" id="pw">—</div></div>
    </div>

    <div class="gripper-bar">
      <span class="gripper-dot open" id="gdot"></span>
      <span id="gtxt">夹爪: 打开 (840)</span>
    </div>

    <div class="key-group-label">平移 XYZ</div>
    <div class="key-hint">
      <div class="key-row"><div class="key" id="kq">Q↑</div><div class="key" id="kw">W→</div><div class="key" id="ke">E↓</div></div>
      <div class="key-row"><div class="key" id="ka">A←</div><div class="key" id="ks">S←</div><div class="key" id="kd">D→</div></div>
    </div>
    <div class="key-group-label">旋转 RPY</div>
    <div class="key-hint">
      <div class="key-row"><div class="key" id="ku">U</div><div class="key" id="ki">I</div><div class="key" id="ko">O</div></div>
      <div class="key-row"><div class="key" id="kj">J</div><div class="key" id="kk">K</div><div class="key" id="kl">L</div></div>
    </div>
    <div class="key-group-label">功能</div>
    <div class="key-hint">
      <div class="key-row">
        <div class="key wide" id="kspace">Space 夹爪</div>
        <div class="key" id="kr">R</div>
      </div>
    </div>
  </div>

</div>

<div class="footer">
  步长: STEP_POS mm / STEP_ROT °  |  Esc 急停
</div>

<script>
const KEY_IDS = {
  'q':'kq','w':'kw','e':'ke','a':'ka','s':'ks','d':'kd',
  'i':'ki','k':'kk','j':'kj','l':'kl','u':'ku','o':'ko',
  'r':'kr',' ':'kspace'
};
const pressedKeys = new Set();

function send(key) {
  fetch('/cmd', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({key})
  })
  .then(r => r.json())
  .then(update)
  .catch(e => {
    document.getElementById('status').textContent = '⚠ 通信失败';
    document.getElementById('status').className = 'warn';
  });
}

function update(d) {
  if (d.level === 'ignore') return;
  const st = document.getElementById('status');
  st.textContent = d.msg;
  st.className = d.level === 'warn' ? 'warn' : '';

  document.getElementById('px').textContent = d.pos.x;
  document.getElementById('py').textContent = d.pos.y;
  document.getElementById('pz').textContent = d.pos.z;
  document.getElementById('pr').textContent = d.pos.roll;
  document.getElementById('pp').textContent = d.pos.pitch;
  document.getElementById('pw').textContent = d.pos.yaw;

  const dot = document.getElementById('gdot');
  dot.className = 'gripper-dot ' + (d.gripper_open ? 'open' : 'close');
  document.getElementById('gtxt').textContent =
    '夹爪: ' + (d.gripper_open ? '打开' : '关闭') + ' (' + d.gripper + ')';
}

// --- Continuous key repeat while held ---
let repeatTimers = {};

document.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'escape') { send('escape'); return; }
  const mapped = k === ' ' ? ' ' : k;
  if (!(mapped in KEY_IDS) && !ACTION_KEYS.has(mapped)) return;
  e.preventDefault();

  // Visual feedback
  const el = KEY_IDS[mapped];
  if (el) document.getElementById(el)?.classList.add('active');

  if (pressedKeys.has(mapped)) return;  // already repeating
  pressedKeys.add(mapped);

  send(mapped);  // immediate first press
  // Start repeat at ~20Hz after 150ms hold
  repeatTimers[mapped] = setTimeout(() => {
    repeatTimers[mapped] = setInterval(() => send(mapped), 50);
  }, 150);
});

document.addEventListener('keyup', e => {
  const k = e.key.toLowerCase();
  const mapped = k === ' ' ? ' ' : k;

  const el = KEY_IDS[mapped];
  if (el) document.getElementById(el)?.classList.remove('active');

  pressedKeys.delete(mapped);
  if (repeatTimers[mapped]) {
    clearTimeout(repeatTimers[mapped]);
    clearInterval(repeatTimers[mapped]);
    delete repeatTimers[mapped];
  }
});

const ACTION_KEYS = new Set('qweasdikjluo'.split(''));

// Replace placeholders in footer
document.querySelector('.footer').innerHTML =
  document.querySelector('.footer').innerHTML
    .replace('STEP_POS', '{{STEP_POS}}')
    .replace('STEP_ROT', '{{STEP_ROT}}');

// Fetch initial state
fetch('/cmd', {method:'POST', headers:{'Content-Type':'application/json'},
  body: JSON.stringify({key:'_init'})}).then(r=>r.json()).then(d => {
    // Just populate the display with current values
    if (d.level !== 'ignore') update(d);
  });
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    """Serve HTML page, MJPEG camera streams, and /cmd POST."""

    def log_message(self, fmt, *args):
        pass  # silence default logs

    def do_GET(self):
        if self.path == "/stream/arm":
            self._stream_mjpeg("arm")
        elif self.path == "/stream/fix":
            self._stream_mjpeg("fix")
        elif self.path == "/":
            self._serve_html()
        else:
            self.send_error(404)

    def _serve_html(self):
        page = HTML_PAGE.replace("{{STEP_POS}}", str(STEP_POS)) \
                         .replace("{{STEP_ROT}}", str(round(math.degrees(STEP_ROT), 2)))
        if args_global.mock:
            page = page.replace(
                '<div id="mode"></div>',
                '<div id="mode"><span class="mock-badge">MOCK 模式</span></div>'
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode())

    def _stream_mjpeg(self, cam_name: str):
        """Send an endless MJPEG stream (multipart/x-mixed-replace)."""
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        try:
            while True:
                with cam_lock:
                    frame = cam_frames.get(cam_name)
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.033)  # ~30fps cap
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected — normal for MJPEG

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        key = body.get("key", "")

        if key == "_init":
            # Return current state without moving
            result = _status("ok", "已连接")
        else:
            result = handle_key(key)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

args_global = None

def main():
    global env, obs, args_global, cam_threads

    parser = argparse.ArgumentParser(description="XArm 键盘控制 Web Demo")
    parser.add_argument("--mock", action="store_true",
                        help="使用模拟环境 (无需真机)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Web 服务端口 (默认 8080)")
    parser.add_argument("--step-pos", type=float, default=2.0,
                        help="平移步长 mm (默认 2.0)")
    parser.add_argument("--step-rot", type=float, default=0.005,
                        help="旋转步长 rad (默认 0.005 ≈ 0.29°)")
    args = parser.parse_args()
    args_global = args

    global STEP_POS, STEP_ROT, ACTION_MAP
    STEP_POS = args.step_pos
    STEP_ROT = args.step_rot
    # Rebuild action map with updated steps
    ACTION_MAP.update({
        "a": np.array([-STEP_POS, 0, 0, 0, 0, 0]),
        "d": np.array([STEP_POS, 0, 0, 0, 0, 0]),
        "w": np.array([0, STEP_POS, 0, 0, 0, 0]),
        "s": np.array([0, -STEP_POS, 0, 0, 0, 0]),
        "q": np.array([0, 0, STEP_POS, 0, 0, 0]),
        "e": np.array([0, 0, -STEP_POS, 0, 0, 0]),
        "i": np.array([0, 0, 0, STEP_ROT, 0, 0]),
        "k": np.array([0, 0, 0, -STEP_ROT, 0, 0]),
        "j": np.array([0, 0, 0, 0, STEP_ROT, 0]),
        "l": np.array([0, 0, 0, 0, -STEP_ROT, 0]),
        "u": np.array([0, 0, 0, 0, 0, STEP_ROT]),
        "o": np.array([0, 0, 0, 0, 0, -STEP_ROT]),
    })

    # --- Init robot env ---
    if args.mock:
        env = MockXArmEnv()
    else:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from xarm_toolkit.env.xarm_env import XArmEnv
        env = XArmEnv(
            action_mode="delta_eef",
            use_force=False,
            initial_gripper_position=840,
        )

    obs = env.reset(close_gripper=False)

    # --- Init cameras ---
    if args.mock:
        cam_frames["arm"] = _make_placeholder_frame("ARM CAM")
        cam_frames["fix"] = _make_placeholder_frame("FIX CAM")
    else:
        from xarm_toolkit.env.realsense_env import RealsenseEnv

        for cam_name, serial in [("arm", "327122075644"), ("fix", "f1271506")]:
            try:
                rs = RealsenseEnv(serial=serial, mode="rgb")
                t = CameraThread(cam_name, rs)
                cam_threads.append(t)
                t.start()
                print(f"  📷 {cam_name.upper()} 相机已启动 (serial={serial})")
            except Exception as e:
                print(f"  ⚠️  {cam_name.upper()} 相机初始化失败: {e}")
                cam_frames[cam_name] = _make_placeholder_frame(
                    f"{cam_name.upper()} OFFLINE"
                )

    # --- Start server ---
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n  🤖 XArm 键盘控制 Demo 已启动")
    print(f"  📡 打开浏览器: http://localhost:{args.port}")
    print(f"  📐 步长: {STEP_POS} mm / {round(math.degrees(STEP_ROT), 2)}°")
    print(f"  {'⚠️  MOCK 模式 — 不连接真机' if args.mock else '✅ 真机模式'}")
    print(f"  按 Ctrl+C 退出\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭 ...")
        server.shutdown()
        for t in cam_threads:
            t.stop()
        env.cleanup()


if __name__ == "__main__":
    main()
