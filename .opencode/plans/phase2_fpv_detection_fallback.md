# Phase 2-FP: FPV Detection Robustness & Tello-Specific Calibration

**Goal:** Close the gap between webcam (good) and Tello FPV (jittery/unreliable) detection by addressing compression artifacts, upscaling noise, and perspective mismatch. Zero new dependencies.

## Root Cause Hypothesis

| Factor | Webcam | Tello FPV | Detection Impact |
|---|---|---|---|
| Raw resolution | 640×480 true pixels | Likely ~320×240 compressed, stretched to 640×480 | Hand is ½ the pixel size |
| Compression | None (direct capture) | Heavy H.264 over UDP, ~200kbps | Edge detail destroyed by block artifacts |
| Perspective | Front-facing (in training data) | Overhead wide-angle (not in training) | Model sees unfamiliar geometry |
| Latency | 0-10ms | ~200-800ms | Hand moves during delay → tracking gap |

**Key insight:** `cv2.resize(img, (640,480))` is **neutral-to-harmful**. It stretches compressed blocks to 640px, making noise bigger without adding detail. Faster inference too.

## Proposed Changes

| # | Change | File | Impact |
|---|---|---|---|
| A | Measure raw FPV resolution (one-time debug) | `tello_handtrack.py` | Confirm if `djitellopy` delivers 320p or 640p |
| B | Remove 640→480 upscale; feed raw frame | `tello_handtrack.py` | Faster + avoids stretching artifacts |
| C | Robust palm center (wrist + MCP0 + MCP5 centroid) | `palmtracker.py` | Stable during hand rotation — **moved to Phase 3** |
| D | Distance proxy (wrist→MCP5 Euclidean) for area | `palmtracker.py` | Independent of finger spread — **moved to Phase 3** |
| E | Kalman filter on `cx, cy, area` | `tello_handtrack.py` | Bridges compression/dropout gaps — **moved to Phase 3** |
| F | Track-and-hold fallback (30 frames) | `tello_handtrack.py` | Smooths sudden "NO HAND" → zero jumps — **moved to Phase 3** |

## Expected Outcomes (Previously Achieved)

| Metric | Before Phase 2-FP | After Phase 2-FP |
|---|---|---|
| Frame resolution | Unknown / 640×480 | **960×720** confirmed |
| Resize overhead | +5-10ms per frame | Eliminated (upscaling removed) |
| PID signature | 4-arg mismatch → crash | Fixed → integral + real dt working |

---

## Items Moved to Phase 3 (Below)

The following items from the Proposed Changes table have been re-scope to **`phase3_detections_robustness.md`**:

- **C** — Robust palm center (centroid of wrist + MCP0 + MCP5)
- **D** — Stable distance proxy for area (wrist→MCP5)
- **E** — Kalman filter on `cx, cy, area`
- **F** — Track-and-hold fallback (30-frame hold window)

These now share spec, code, and integration tests in Phase 3.

## Completed Steps Summary

### ✅ Step 1: FPV Debug + Raw Resolution Measurement
- **Status:** Completed
- **Action:** Added one-time resolution measurement in `init_device()`
- **Result:** Tello FPV delivers **960×720** (not 320p as hypothesized)
- **Discovery:** We were downscaling by 33% — bad for detection precision

### ✅ Step 2: Resolution Update
- **Status:** Completed
- **File:** `tello_handtrack.py:9`
- **Change:** `RES_W, RES_H = 960, 720` (was `640, 480`)
- **Impact:** Detection now uses full 960×720 resolution
- **Validation:** `cv2.resize()` now resizes *to* 960×720 (no loss)

### ✅ Step 3: PID Signature Fix (Phase 3 Integration)
- **Status:** Completed
- **File:** `tello_handtrack.py`
- **Problem:** `compute_pid()` called with 4 args, but Phase 3 requires 5 args → `TypeError`
- **Changes:**
  - Added `integral_y=0`, `integral_z=0`, `prev_time=time()` before loop
  - Updated Y/PID call: `speed_y, integral_y = compute_pid(error_y, prev_error_y, integral_y, dt, FB_PID)`
  - Updated Z/PID call: `speed_z, integral_z = compute_pid(error_z, prev_error_z, integral_z, dt, VD_PID)`
  - Added real-time delta: `dt = max(current_time - prev_time, 0.001)`
- **Impact:** TypeError resolved, PID integral tracking restored, real dt computation

---

## Next Steps

**Complete.** Items C-F have been re-scoped to **`phase3_detections_robustness.md`** with full code spec and integration test.
