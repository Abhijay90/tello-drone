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
| `vision/facetracker.py` | Face detection + PID controller (`findFace`, `trackface`) |
| `vision/video_stream.py` | Matplotlib-based video stream class |

Always run with a real Tello connected to the same Wi-Fi network. No simulator mode.

## Key constraints
- Every script calls `me.connect()` inline — no shared drone instance or config.
- Face detection loads the cascade from relative path `vision/haarcascade_frontalface_default.xml` — run from repo root.
- `keyboard_control.py` has a key-press helper but is unfinished (typos, no drone wiring).
- `requirement.txt` is misspelled (`requirement.txt` not `requirements.txt`) — install with that exact filename.
- No `__pycache__`, `*.egg-info`, environments, or logs are tracked.

## Architecture notes
- Single package, no imports beyond stdlib + deps. All logic in flat scripts.
- `vision/` is a directory (not a package — no `__init__.py`). Import path in `tello_facetrack.py` uses `vision.facetracker` which works from repo root.
