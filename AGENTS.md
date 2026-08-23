# Tello Drone - AGENTS.md

## Setup
```
pip install -r requirement.txt
```
Depends: `djitellopy`, `opencv-python==4.12.0.88`, `matplotlib`, `pygame`, `numpy`.

## Running
All entrypoints are standalone scripts — no test framework, no linter, no build step.

| Script | What it does |
|---|---|
| `tello.py` | Basic flight: takeoff, rc-movement sequence, land |
| `video.py` | Live FPV stream via cv2 window |
| `drone_flip.py` | Takeoff to 100cm, flip back, land |
| `tello_facetrack.py` | PID face-tracking flight loop |
| `drone_controller.py` | Central drone movement layer (`DroneController`) — reusable by any control system |
| `drone_control.py` | Unified flight control: auto-detects gamepad (cine-style sticks) or keyboard fallback, hot-plug safe, cv2 FPV window |
| `vision/facetracker.py` | Face detection + PID controller (`findFace`, `trackface`) |
| `vision/video_stream.py` | Matplotlib-based video stream class |

Always run with a real Tello connected to the same Wi-Fi network. No simulator mode.

## Key constraints
- Most scripts call `me.connect()` inline; `keyboard_control.py` uses the shared `DroneController` (`drone_controller.py`) instead.
- Face detection loads the cascade from relative path `vision/haarcascade_frontalface_default.xml` — run from repo root.
- `drone_control.py` drives `drone_controller.DroneController` with auto-detected input: gamepad (cine-style sticks, full-stick speed = `GAMEPAD_MAX_VELOCITY`, default 50 cm/s; A takeoff / B land / Start emergency) if connected, else keyboard (WASD+R/F/Q/E, SPACE/L/ESC). Hot-plug safe — switching input mid-flight makes the drone hover. Keyboard needs the pygame window focused; gamepad does not.
- `requirement.txt` is misspelled (`requirement.txt` not `requirements.txt`) — install with that exact filename.
- No `__pycache__`, `*.egg-info`, environments, or logs are tracked.

## Architecture notes
- Single package, no imports beyond stdlib + deps. All logic in flat scripts.
- `vision/` is a directory (not a package — no `__init__.py`). Import path in `tello_facetrack.py` uses `vision.facetracker` which works from repo root.
- 2-file control pattern: `drone_controller.py` holds all drone movement (no input deps); input sources (keyboard now, gestures later) map onto its API. See PROJECT_REFERENCE.md "Keyboard Control".
