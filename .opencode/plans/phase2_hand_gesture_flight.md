# Phase 2: Hand Gesture Flight Control

**Goal:** Create `tello_handtrack.py`, a flight-enabled script that maps palm center position and bounding box area to drone RC commands via real-time PD control.

## 📁 New Files
| File | Purpose |
|---|-|
| `tello_handtrack.py` | Main flight control loop (Tello RC commands + Palm PID) |

## ⚙️ `tello_handtrack.py` Specification

### Architecture
Follows `tello_facetrack.py` structure but splits horizontal (pan) and vertical (altitude) control into independent PD loops.

### Configuration
- **Frame size:** `640 x 480` (resized Tello stream) — increased from 360x240 for better Mediapipe hand detection
- **Tracking:** `vision.palmtracker.findPalm()` → returns `img`, `(cx, cy, area)`
- **PID Gains:** `pid_y = [0.4, 0.4, 0]` (forward/back), `pid_z = [0.4, 0.4, 0]` (up/down)
- **Dead zone:** `20` (stops PID commands when error drops below this threshold)
- **Target distance:** `target_area = 5000` (approximates 1m distance to hand)

### Control Logic Mapping
| Hand State | Error Calculation | Drives | Result |
|---|---|---|---|
| Palm left of center | `error_y = cx - 180` (positive) | `y` axis forward | Drone pushes forward |
| Palm right of center | `error_y` (negative) | `y` axis backward | Drone pulls back |
| Hand moves away (area decreases) | `error_z = target_area - area` (positive) | `z` axis up | Drone climbs |
| Hand moves closer (area increases) | `error_z` (negative) | `z` axis down | Drone descends |
| Within dead zone | `abs(error) < 20` | Hover command `(0,0,0,0)` | Drone stabilizes |
| Hand lost | `cx = 0` | Hover command | Drone stabilizes |

### Function-Level Flow
```python
# 1. Setup
me = Tello(); me.connect(); me.streamon(); me.takeoff()
w, h = 640, 480  # Increased from 360x240 for hand detection

# 2. Main Loop
while True:
    img = me.get_frame_read().frame       # Raw Tello frame
    print('Raw frame shape:', img.shape)   # Debug logging
    img = cv2.resize(img, (w, h))         # Resize to 640x480
    img, (cx, cy, area) = findPalm(img)
    
    # Y-Axis Control (Forward/Back)
    error_y = cx - w // 2
    speed_y = compute_pid(error_y, prev_error_y, pid_y) if abs(error_y) > 20 else 0
    
    # Z-Axis Control (Altitude via Area)
    error_z = target_area - area
    speed_z = compute_pid(error_z, prev_error_z, pid_z) if abs(error_z) > 20 else 0
    
    # Send RC & Display
    me.send_rc_control(0, speed_y, speed_z, 0)
    cv2.imshow("Palm Flight", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    prev_error_y, prev_error_z = error_y, error_z

# 3. Cleanup
me.streamoff(); me.land()
```

### 📷 Camera Stream Test

**Purpose:** Pre-flight verification that Tello FPV stream is reachable and delivers valid frames before any takeoff or RC commands.

**Behavior:**
- Calls `me.streamon()` and reads 3 frames from `me.get_frame_read().frame`
- Validates each frame is non-None and `frame.size > 0`
- Measures per-frame latency via `time.time()` delta
- Reports: frames received / 3, average latency, estimated FPS
- If all 3 succeed: displays success in overlay window, waits for keypress
- If any fail: prints error with troubleshooting hint, returns False

**Placement in `tello_handtrack.py`:**
```python
# After me.connect():
ok, _ = test_drone_camera(me)
if not ok:
    sys.exit(1)
me.streamoff()
me.streamon()       # Start flight stream
me.takeoff()
```

**Success output:**
```
  Frame 1: latency=24ms
  Frame 2: latency=22ms
  Frame 3: latency=23ms
  ✅ Stream OK: 3/3 frames, avg=23ms, ~43 FPS
  Press any key to takeoff...
```

**Failure output:**
```
  Frame 1: latency=0ms
  Frame 2: latency=0ms
  Frame 3: latency=0ms
  ❌ Stream FAILED: only 0/3 frames received.
  Check Wi-Fi, Tello power, and that you are on the same network.
```

---

## ✅ Phase 2 Execution Summary

### Files Created
| Status | File | Role |
|---|-|-|
| ✅ done | `tello_handtrack.py` | Main flight control (palm PID + RC) |

### Configuration & Decisions
| Parameter | Value | Notes |
|---|---|---|
| `RES_W, RES_H` | **640, 480** | Resized frame dimensions (increased from 360x240) |
| `TARGET_AREA` | **4000** | Lower default (~closer to drone). Tune in Phase 3. |
| `DEAD_ZONE` | 20 | Stops commands when error < threshold |
| `FB_PID` | [0.4, 0.4, 0] | Forward/back control (X-axis) |
| `VD_PID` | [0.4, 0.0, 0.2] | Up/down control (Z-axis). D-term 0.2 dampens jitter. |
| `TARGET_X` | **320** | Frame center (RES_W // 2) |
| `DRONE_MODE` | **True** | Toggle drone vs webcam debug mode (line 22) |
| `WEBCAM_IDX` | 0 | Webcam source index when DRONE_MODE=False (line 25) |

### Control Mapping (Confirmed)
| Input from `findPalm()` | Interpretation | Output Command | Drone Behavior |
|---|---|-|
| `cx > 320` | Palm on right half | `speedY` (negative, bounded ±50) | Drone moves **backward** |
| `cx < 320` | Palm on left half | `speedY` (positive, bounded ±50) | Drone moves **forward** |
| `area < 4000` | Hand too far | `speedZ` (positive, bounded ±50) | Drone **climbs** |
| `area > 4000` | Hand too close | `speedZ` (negative, bounded ±50) | Drone **descends** |
| `cx == 0` | No hand detected | `(0,0,0,0)` | **Hover** in place |
| `abs(error) < 20` | Within dead zone | `0` | **Hover** (prevents jitter) |

### Tuning Decisions (Deferred to Phase 3)
- **[ ]** Control polarity — Default (not flipped). Will invert if drone moves opposite of palm during first flight test.
- **[ ]** Damping (D-term) — Kept `VD_PID = [0.4, 0.0, 0.2]`. Will revert to pure P `[0.4, 0, 0]` if D-term causes instability.
- **[ ]** `TARGET_AREA` — Will adjust ±10% from 4000 based on real altitude hold behavior.
- **[ ]** `FB_PID` gains — May increase Kp if forward response is sluggish.

### Files Summary
| Status | File | Role |
|---|-|---|
| ✅ Existing | `vision/palmtracker.py` | Palm detection (Phase 1, Task API migrated) |
| 🏗️ **NEW** | `tello_handtrack.py` | Main flight loop + camera test |
| 📷 **INLINE** | `test_drone_camera()` | Pre-flight FPV stream test |
| 📖 Reference | `tello_facetrack.py` | Architectural template |

### Implementation Checklist
- [x] Create `tello_handtrack.py` with camera test
- [x] Add `test_drone_camera()` with per-frame overlay feedback
- [x] Add `DRONE_MODE` toggle (line 22) — True for real drone, False for webcam debug
- [x] Add `WEBCAM_IDX` config (line 25) — webcam source for debug mode
- [ ] Test on drone (check polarity before flight)
- [ ] Tune `TARGET_AREA` in Phase 3

### Implementation Details
| Feature | Location | Description |
|---|-|-|
| Mode toggle | `tello_handtrack.py:22` | `DRONE_MODE = True` switches between drone/stream and webcam debug |
| Webcam source | `tello_handtrack.py:25` | `WEBCAM_IDX = 0` — change index for different camera |
| Unified bootstrap | `tello_handtrack.py:98-143` | `init_device()` returns `(me, cap)` tuple based on mode |
| Mode-gated RC | `tello_handtrack.py:148-151` | `send_rc()` only fires commands in DRONE_MODE |
| Mode-gated frames | `tello_handtrack.py:154-162` | `get_frame()` pulls from drone stream or webcam |
| Mode badge | `tello_handtrack.py:212` | `[DRONE MODE]` / `[WEBCAM MODE]` overlay on flight window |
| Graceful shutdown | `finally` block:233 | Drone mode → `streamoff()` + `land()`; Webcam mode → `cap.release()` |

## 🔜 Phase 3 Preview: Tuning & Hardening
- Run flight tests to tune `target_area` and refine `pid_z` gains (vertical lift usually requires higher Kp than horizontal).
- Add graceful drift handling: what should the drone do if tracking drops for >10 frames before hovering?
- Add battery monitoring and auto-land warnings.
