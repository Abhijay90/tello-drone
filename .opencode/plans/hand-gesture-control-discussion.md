# Hand Gesture Drone Control - Ongoing Discussion

**Status:** Planning phase — not yet implemented
**Decided:** Continue later from this point

---

## Goal
Replace face detection with palm detection in a Tello drone control script. Use hand position in the camera frame to control drone movement (forward, backward, up, down).

## Architecture Decisions

### Detection: Haar Cascade + Skin Color (Not MediaPipe)
- **Why:** Lower CPU load (~10-15% i5 vs 15-30%), no new dependencies, >60fps achievable at 360x240
- **Method:** HSV skin-color segmentation + OpenCV built-in hand cascade (with skin-color-only fallback)
- **Skin color HSV ranges (OpenCV BGR→HSV):**
  - Lower: `(0, 48, 0)` — Upper: `(20, 255, 255)`
  - Lower: `(160, 48, 0)` — Upper: `(200, 255, 255)`
- **Combine:** When both detectors produce boxes, merge via overlap/union; fall back to whichever finds something

### Control: Palm Relative Position (Continuous)
Same pattern as the existing `tello_facetrack.py` — palm center position in frame maps to drone RC commands, not finger counting.

| Palm Position | RC Command | Movement |
|---|---|---|
| Left of frame center | `+speed` on y-axis | Forward |
| Right of frame center | `-speed` on y-axis | Backward |
| Bottom of frame center | `-speed` on z-axis | Up |
| Top of frame center | `+speed` on z-axis | Down |
| Near center | Hover (no movement) | Hold position |
| No hand detected | Hover (no movement) | Hold position |

### PID Strategy: Separate Gains for Each Axis
- **Separate PID values** for `y` (forward/back) and `z` (up/down)
- **Why:** Drone physics differ — horizontal settling is faster with less drift, vertical fights gravity and has more overshoot. Same PID makes both suboptimal.
- Starting gain values: same as face tracker `[0.4, 0.4, 0]` for each axis independently

## File Changes

### New Files
| File | Purpose |
|---|---|
| `tello_handtrack.py` | Main entrypoint script |
| `vision/palmtracker.py` | `findPalm()` + `trackPalm()` functions |

### Existing Files to Reference
| File | Reference for |
|---|---|
| `tello_facetrack.py` | Main loop structure |
| `vision/facetracker.py` | findFace/trackface → findPalm/trackPalm pattern |
| `tello.py` | Tello connect/takeoff/send_rc_control/land sequence |
| `requirement.txt` | No new dependencies needed |

## Implementation Plan (Not yet started)

### 1. `vision/palmtracker.py`

```
def findPalm(img):
    # 1. HSV skin-color mask via cv2.inRange
    # 2. Contour detection on mask
    # 3. Haar cascade (cv2.haarcascades_data/haarcascade_hand.rxml)
    # 4. Merge bounding boxes from both sources
    # 5. Draw rectangle + center circle on img
    # 6. Return (img, [center_x, center_y, area]) or ([[0,0],0])
```

### 2. `tello_handtrack.py`

```python
from djitellopy import tello
from time import sleep
from vision.palmtracker import findPalm, trackPalm

me = tello.Tello()
me.connect()
me.streamon()
me.takeoff()

w, h = 360, 240
pid_y = [0.4, 0.4, 0]  # forward/back separate
pid_z = [0.4, 0.4, 0]  # up/down separate
pError_y, pError_z = 0, 0

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findPalm(img)
    trackPalm(me, info, w, h, pid_y, pid_z, pError_y, pError_z)
    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

me.streamoff()
me.land()
```

### 3. `requirement.txt`
No changes — uses existing deps (`cv2`, `numpy`, `djitellopy`).

## Key Constraints
- Every script calls `me.connect()` inline — no config file
- Run from repo root (relative path `vision/` cascades)
- No `__init__.py` in `vision/` — not a package
- Requires a real Tello on the same Wi-Fi network

---

**Ready to implement when you're ready. Say "implement the plan above" or add modifications.**
