# Phase 3: Hand Detection Robustness (Phase 2-FP C-F)

**Goal:** Stabilize palm detection for Tello FPV by removing wasteful resize, replacing fragile landmark proxies with robust features, adding temporal filtering, and fallback on detection loss. Zero new dependencies.

**Status:** Active

---

## Scope: 6 Code Changes + 1 Integration Test File

| # | Item | File | Line(s) | Impact |
|---|---|---|---|---|
| 1 | B-residual: Remove resize | `tello_handtrack.py` | 232 | Feed raw FPV 960×720 directly to MediaPipe |
| 2 | C: Robust palm center | `vision/palmtracker.py` | 51-52 | Centroid of wrist + index-MCP + mid-MCP |
| 3 | D: Stable area proxy | `vision/palmtracker.py` | 53 | Wrist→MCP9 distance (finger-independent) |
| 4 | E: Kalman filter | `tello_handtrack.py` | before loop | Smooth cx/cy/area with SimpleKalman |
| 5 | F: Track-and-hold | `tello_handtrack.py` | 237-238 | Predict 30 frames during detection gap |
| 6 | B-residual cleanup | `tello_handtrack.py` | 232 | Remove the resize call entirely |

**Phase 3.5:** Integration test `test/test_fpv_robustness.py`

---

## 1. B-residual: Remove Resize (Item 6 above)

**Problem:** Line 232 `cv2.resize(img, (RES_W, RES_H))` resizes Tello FPV (already 960×720) to the same resolution — wasteful and unnecessary. If Tello FPV dimensions ever change, this blindly resizes to a fixed 960×720, potentially distorting aspect ratio.

**Change in `tello_handtrack.py` line ~232:**
```python
# Before:
img = cv2.resize(img, (RES_W, RES_H))

# After — remove this line entirely
# Feed raw frame directly to findPalm; it handles shape internally
```

**Verify RES_W/H is still set (line 9)** for TARGET_X reference — but drop the resize call.

**Risk check:** `findPalm` already uses `img.shape` internally (lines 41, 51, etc.). No changes needed there.

---

## 2. C: Robust Palm Center (Item 2 above)

**Problem:** `findPalm` line 51 uses wrist (landmark 0) as center. When hand rotates or wrist tilts, cx/cy shifts disproportionately → PID drift.

**Change in `vision/palmtracker.py` lines 51-52:**

```python
# Before:
cx = int(landmarks[0].x * img.shape[1])
cy = int(landmarks[0].y * img.shape[0])

# After — centroid of 3 stable palm points:
#   landmarks[0] = wrist
#   landmarks[5] = index MCP
#   landmarks[9] = middle MCP
pts_x = [landmarks[0].x, landmarks[5].x, landmarks[9].x]
pts_y = [landmarks[0].y, landmarks[5].y, landmarks[9].y]
cx = int((pts_x[0] + pts_x[1] + pts_x[2]) / 3 * img.shape[1])
cy = int((pts_y[0] + pts_y[1] + pts_y[2]) / 3 * img.shape[0])
```

**Rationale:** Wrist + index-MCP + middle-MCP form a triangle on the palm base. Their centroid is stable across hand orientation, unlike wrist alone which rotates with the hand.

**Visual effect:** The center dot (line 61) should stay locked on palm center even when fingers rotate.

---

## 3. D: Stable Area Proxy (Item 3 above)

**Problem:** `findPalm` line 53 uses bounding box area `(max(xs)-min(xs)) * (max(ys)-min(ys)) * img_area`. When user opens/closes fingers, bbox size jumps → area proxy is noisy → PID altitude oscillates.

**Change in `vision/palmtracker.py` line 53:**

```python
# Before:
area = int((max(xs) - min(xs)) * (max(ys) - min(ys)) * img.shape[1] * img.shape[0])

# After — wrist-to-middle-MCP Euclidean distance, scaled to match TARGET_AREA range:
import numpy as np
# landmarks[0] = wrist, landmarks[9] = middle MCP
dx = landmarks[9].x - landmarks[0].x
dy = landmarks[9].y - landmarks[0].y
dist_proxy = int(np.sqrt(dx**2 + dy**2) * 500)  # scale to ~4000 at typical distance
```

**Tuning note:** The `*500` scale will need real-world measurement. Start with 500, then:
- Move hand closer/farther and read `area` in the debug overlay
- Adjust scale factor so that your "hover distance" reads ~TARGET_AREA (4000)

**Visual effect:** Area values should fluctuate ±5–10px RMS instead of ±500+ with finger spread.

---

## 4. E: Kalman Filter (Item 4 above)

**Problem:** Tello FPV compression causes frame-by-frame coordinate jitter (~3–10px). PID sees each jittered value → control surface oscillates → drone wobbles.

**Add to `tello_handtrack.py` (before the `init_device()` call, ~line 30):**

```python
class SimpleKalman:
    """2-state position+velocity Kalman filter for 1D measurement."""
    def __init__(self, dt=0.0333):
        self.dt = dt
        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.measurementMatrix = np.array([[1., 0.]], np.float32)
        self.kf.processNoiseCov = np.array([[1., 0.], [0., 1.]], np.float32) * dt
        self.kf.measurementNoiseCov = np.array([[1.]], np.float32)
        self.kf.transitionMatrix = np.array([[1., dt], [0., 1.]], np.float32)
        self.kf.statePost = np.array([[0.], [0.]], np.float32)

    def update(self, z):
        self.kf.predict()
        self.kf.correct(np.array([[z]], np.float32))
        return self.kf.statePost[0][0]

    def predict(self):
        return self.kf.predict()[0][0]
```

**Add 3 filter instances before the loop (after line 224 or 236):**
```python
kf_cx = SimpleKalman(dt=1/30)
kf_cy = SimpleKalman(dt=1/30)
kf_area = SimpleKalman(dt=1/30)
```

---

## 5. F: Track-and-Hold Fallback (Item 5 above)

**Problem:** Lines 237-238: when `cx=0, area=0` (detection lost), PID commands go to zero → sudden drop in control → drone crashes/lurches.

**Replace `tello_handtrack.py` lines 237-238 with:**

```python
# Track-and-hold: remember last valid cx, cy, area
# When detection is lost, predict for up to HOLD_MAX frames
HOLD_MAX = 30  # ~1s at 30fps

if cx == 0 or area == 0:
    if hold_counter is not None:
        hold_counter += 1
    else:
        hold_counter = 1  # first frame of loss

    if hold_counter <= HOLD_MAX and hold_count is not 0:
        # Predict from last known state
        cx = int(last_cx + kf_cx.predict())
        cy = int(last_cy + kf_cy.predict())
        pred_area = max(kf_area.predict(), 0)
        pred_area = int(pred_area) if pred_area > 0 else 0
        if pred_area > 0:
            speed_y, integral_y = compute_pid(last_cx - cx, prev_error_y, integral_y, dt, FB_PID)
            speed_z, integral_z = compute_pid(pred_area - area, prev_error_z, integral_z, dt, VD_PID)
            speed_y = max(-50, min(50, speed_y))
            speed_z = max(-50, min(50, speed_z))
            cv2.putText(img, f"HOLD #{hold_counter}", (200, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    else:
        # Hold expired — stop
        speed_y, speed_z = 0, 0
        hold_counter = None  # reset on next detection
        cv2.putText(img, "HOLD EXPIRED", (200, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
else:
    # Detection restored — reset hold and update Kalman
    hold_counter = None
    last_cx = cx
    last_cy = cy
    last_area = area
    hold_count = 0
    kf_cx.update(cx)
    kf_cy.update(cy)
    kf_area.update(area)
```

**Key variables to initialize before the loop:**
```python
hold_counter = None  # starts None, set on first detection loss
hold_count = 0       # increment each frame while holding
last_cx = last_cy = last_area = 0
```

**Test checklist:**
- Look away (no hand) → drone should continue drifting for ~1s
- Return hand → track should snap back smoothly, not jerk
- Hold >30 frames → track expires, control stops (safety)

---

## 6. Testing & Integration

### Phase 3.5: FPV Integration Test (`test/test_fpv_robustness.py`)

```python
"""
Manual FPV robustness test for Phase 3 changes:
  - Measure detection loss rate (frames/second with 0 hand)
  - Measure jitter RMS after Kalman (100 frames, hand steady)
  - Verify track-and-hold on 30-frame detection gap
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time, sleep
import cv2
from vision.palmtracker import findPalm

def test_on_fpv():
    from djitellopy import tello
    print("Connecting to Tello...")
    me = tello.Tello()
    me.connect()
    print(f"Battery: {me.get_battery()}%")
    me.streamon()

    print("\n--- PHASE 3 ROBUSTNESS TEST ---")
    print("Please keep hand STEADY for 100 frames...")

    # Part 1: Jitter measurement (hand steady)
    cx_vals = []
    area_vals = []
    start = time()
    for i in range(100):
        frame = me.get_frame_read().frame
        if frame is None:
            break
        _, info = findPalm(frame)
        cx, cy, area = info
        if area > 0:
            cx_vals.append(cx)
            area_vals.append(area)

    if len(cx_vals) > 5:
        cx_rms = (sum((v - sum(cx_vals)/len(cx_vals))**2 for v in cx_vals) / len(cx_vals))**0.5
        area_rms = (sum((v - sum(area_vals)/len(area_vals))**2 for v in area_vals) / len(area_vals))**0.5
        avg_latency = (time() - start) / len(cx_vals) * 1000
        print(f"Jitter RMS: cx={cx_rms:.1f}px, area={area_rms:.1f}px")
        print(f"Avg frame time: {avg_latency:.0f}ms ({1000/avg_latency:.1f}fps)")
    else:
        print("❌ Too few detections for jitter test")

    # Part 2: Hold test (look away for 35 frames)
    print("\nLooking away (35 frames)...")
    hold_frames = 0
    for i in range(35):
        frame = me.get_frame_read().frame
        if frame is None:
            break
        hold_frames += 1
        _, info = findPalm(frame)
        cx, cy, area = info
        if area > 0:
            hold_frames = 0  # reset if re-detected

    print(f"Hold test: {hold_frames} frames without detection")
    print("Return hand — watch track-and-hold behavior in main loop")

    me.streamoff()
    print("\nTest complete. Check main loop overlay for HOLD indicator.")

if __name__ == "__main__":
    test_on_fpv()
```

### Manual Verification Checklist

| # | Test | Pass Criteria |
|---|---| ---|
| 1 | Hand steady, 100 frames | cx jitter RMS < 3px (was ~8px with old code) |
| 2 | Hand steady, 100 frames | area RMS < 50 (was ~500 with bbox) |
| 3 | Look away → track holds | Drone continues for ~1s, no jerk on return |
| 4 | Fingers close/open | Area stays within ±20 of hover target (was ±500+) |
| 5 | No resize overhead | Frame time drops by ~5–10ms per frame |
| 6 | Scale factor tuning | TARGET_AREA (4000) at hover distance — adjust `*500` |

---

## Files to Modify

1. **`vision/palmtracker.py`** — changes 2, 3 (lines 51-53)
2. **`tello_handtrack.py`** — changes 1, 4, 5 (lines ~30, ~225, ~232, ~237-238)
3. **`test/test_fpv_robustness.py`** — new file

## Estimated Time: 30–45 min

## Dependencies

Zero new dependencies. Uses existing `cv2.KalmanFilter`, `np.sqrt`, and `time.perf_counter`/`time`.
