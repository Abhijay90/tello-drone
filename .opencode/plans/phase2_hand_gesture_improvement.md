# Phase 2: Hand Gesture Improvement — Plan

- Separate gesture detection into its own module (`vision/gestures.py`)
- Create standalone tester (`test/gesture_tester.py`) with drone-motion overlay
- Update `vision/palmtracker.py` to include raw landmarks for gesture module

---

## Goal

All gesture detection logic lives in its own file, separate from palm tracking.
A tester verifies every gesture and maps it to Tello drone directions.

---

## File Inventory

| File | Action | Responsibility |
|---|---|-::--|
| `vision/gestures.py` | **NEW** | Pure gesture classifier |
| `test/gesture_tester.py` | **NEW** | Interactive overlay tester with drone direction mapping |
| `vision/palmtracker.py` | **EDIT** | Append landmarks to return value |
| `tello_handtrack.py` | **EDIT** | Integrate gesture classifier, drive drone RC commands, add gesture overlay |

No other files are changed. `tello_handtrack.py` is added in this update.

---

## Architecture

```
palmtracker.py findPalm(img) → [cx, cy, area, landmarks]
                                    │
                                    ▼
                            gestures.py classify(landmarks, W, H) → gesture string
                                          │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                     gesture_tester.py          tello_handtrack.py
                     (overlay / preview)        → RC flight commands
                              │                    (gesture → drone map)
                              ▼
                   overlay: gesture name, thumb distance, finger count
```

---

## 1. `vision/gestures.py` — Gesture Classifier

### Constants

| Constant | Value | Measurement |
|---|---|---|
| `THUMB_EXTEND_PX` | 60 | Thumb tip (4) to wrist (0) distance |
| `FINGER_EXTEND_PX` | 30 | Finger tip to MCP joint distance |
| `PALM_ORIENT_OFFSET` | 30 | Wrist Y vs MCP5 Y gap for palm tilt |

### Landmarks Used

| Joint | Index | Role |
|---|---|---|
| Wrist | 0 | Thumb-reference |
| Thumb tip (IP joint) | 4 | Thumb extension |
| MCP thumb | 5 | Palm orientation reference |
| Index tip → MCP | 8 → 5 | Finger 1 |
| Middle tip → MCP | 12 → 9 | Finger 2 |
| Ring tip → MCP | 16 → 13 | Finger 3 |
| Pinky tip → MCP | 20 → 17 | Finger 4 |

### Gesture Definitions

| Name | Thumb | Fingers | Palm Orientation |
|---|---|-:-:--|
| `open_palm` | Extended | All 4 extended | No tilt |
| `closed_fist` | Not extended | All 4 curled | — |
| `thumbs_up` | Extended | All 4 curled | — |
| `thumbs_down` | Not extended | All 4 extended | — |
| `palm_down` | Extended | All 4 extended | Wrist below MCP5 |
| `palm_up` | Extended | All 4 extended | Wrist above MCP5 |

### Classify Logic (Ordered Checks)

```
1. NOT thumb_ext AND all fingers extended  → THUMBS_DOWN
2. thumb_ext AND all fingers curled        → THUMBS_UP
3. thumb_ext AND all fingers extended:
     if wrist_below_mcp: → PALM_DOWN
     elif wrist_above_mcp: → PALM_UP
     else: → OPEN_PALM
4. NOT thumb_ext AND all fingers curled    → CLOSED_FIST
5. else                                  → UNKNOWN
```

### Exports

- `Gesture` class with string constants
- `classify(landmarks, img_w, img_h) → str`
- `get_thumb_distance(landmarks, img_w, img_h) → float`
- `get_finger_count(landmarks, img_w, img_h) → tuple`
- `_euclidean(lm1, lm2, img_w, img_h) → float` (private)

---

### 2. `vision/palmtracker.py` — Changes

#### Change 1: Append landmarks to return (line 66)

```python
# Before:
return img, [cx, cy, area]

# After:
return img, [cx, cy, area, landmarks if detection_result.hand_landmarks else None]
```

#### Change 2: Handle no-hand case (lines 62–65)

```python
# Before:
else:
    cv2.putText(img, "NO HAND", ...)
return img, [cx, cy, area]

# After:
else:
    cv2.putText(img, "NO HAND", ...)
return img, [0, 0, 0, None]
```

---

### 3. `test/gesture_tester.py` — Interactive Tester

#### Constants

```python
DATA_SOURCE = "webcam"  # "webcam" | "drone"
WEBCAM_IDX = 0
```

#### Gesture-Drone Mapping

| Gesture | Arrow Icon | Text Overlay |
|---|---|-:-:--|
| `open_palm` | → | `FORWARD` |
| `closed_fist` | ← | `BACKWARD` |
| `thumbs_up` | ↑ | `UP` |
| `thumbs_down` | ↓ | `DOWN` |
| `palm_down` | ↑ | `DIST HOLD` |
| `palm_up` | HOVER | `HOVER` |
| `unknown` | ○ | `HOVER` |
| No Hand | — | `NO HAND` |

#### Overlay Layout

```
──────────── Top-Bar (3 lines) ─────────
Gesture: open_palm  │ cx:480 cy:360  │ area:4500
THUMB: 92px         │ FINGER: 4/4

──────────── Main Canvas ─────────
[Live Frame W/ Landmarks]          ┌─ GESTURE TABLE ─┐
                                   │ ● open_palm       │
                                   │   → FORWARD       │
                                   │ ○ thumbs_up       │
                                   │   ↑ UP            │
                                   │ ○ thumbs_down     │
                                   │   ↓ DOWN          │
                                   │ ○ palm_down       │
                                   │   ↑ DIST HOLD     │
                                   │ ○ palm_up         │
                                   │   HOVER           │
                                   │ ○ closed_fist     │
                                   │   ← BACKWARD      │
                                   │ ○ unknown         │
                                   │   HOVER           │
                                   └───────────────────┘
──────────── Status Footer ─────────
GESTURE TESTER v2 | Press Q: Exit | T: Toggle Source
```

#### Terminal Output (fallback if no display)

```
FRAME 42: open_palm | cx:480 cy:360 area:4500 | THUMB: 92px │ FINGERS: 4/4 │ → FORWARD
```

#### Keyboard

| Key | Action |
|---|---|
| `q` | Exit |
| `t` | Toggle `DATA_SOURCE` (webcam ↔ drone) |

---

---

## 4. `tello_handtrack.py` — Gesture Flight Integration

tello_handtrack.py will be updated to integrate the gesture classifier, map gestures to drone RC commands, and display gesture overlay info.

### Control Logic Mapping

| Gesture | Drone Command | RC Mapping |
|---|-::--|
| `open_palm` | Forward | `speed_y = +40` |
| `closed_fist` | Backward | `speed_y = -40` |
| `thumbs_up` | Up | `speed_z = -40` |
| `thumbs_down` | Down | `speed_z = +40` |
| `palm_down` / `palm_up` / `unknown` | Hover/Stop | `speed_y = 0, speed_z = 0` |

### Implementation Notes

- Add landmarks to `info` unpacking: `cx, cy, area, landmark = info`
- Call `classify()` when `landmark is not None`
- Replace PID block (lines ~241-261) with gesture switch/map
- Append gesture/THUMB/FINGER text overlay below existing PID text (line ~270)
- Keep `DRONE_MODE` toggle intact
- Remove or comment out existing PID flight logic

---

## Tester Verification Checklist

| Test Pose | Expected Gesture | Expected Overlay |
|---|---|-:-:--|
| Open palm, thumb out | `open_palm` | `→ FORWARD` (highlighted) |
| Closed fist | `closed_fist` | `← BACKWARD` |
| Thumb up only | `thumbs_up` | `↑ UP` |
| Thumb tucked, fingers out | `thumbs_down` | `↓ DOWN` |
| Open palm, wrist below MCP5 | `palm_down` | `↑ DIST HOLD` |
| Open palm, wrist above MCP5 | `palm_up` | `HOVER` |
| Mid-state gesture | `unknown` | `HOVER` |
| No hand | `unknown` | `no_hand.txt` |

---

## Risk & Mitigation

| Risk | Mitigation |
|---|---|
| Gesture jitter at boundaries | `PALM_ORIENT_OFFSET` = 30px hysteresis buffer |
| Different camera distances change thresholds | Document tuning steps; thresholds are module-level constants for easy editing |
| `palmtracker` landmarks change type | Explicit `isinstance` check before calling `classify()` |
| Tello stream drops during test | `get_frame()` returns `None` after 2s timeout, tester skips frame |

---

## Tuning Procedure (if needed)

1. Run tester in webcam mode
2. Show each gesture 5 times consecutively
3. If jitter occurs: increase threshold by +5
4. If gesture never triggers: decrease threshold by −5
5. Re-test until stable

---

## Execution Order (Updated)

| Step | File | Why First |
|---|---|-::--|
| 1 | `vision/gestures.py` | Core logic, no dependencies ✓ |
| 2 | `vision/palmtracker.py` (edit) | Makes landmarks available ✓ |
| 3 | `test/gesture_tester.py` | Consumes modules from steps 1–2 ✓ |
| 4 | `tello_handtrack.py` | Consumes gestures + landmarks for flight ✓ |

Steps 1–3 are parallelizable. Step 4 depends on 1 and 2 being correct.

---

## Tuning Guide (for future)

1. Run `gesture_tester.py` in `DATA_SOURCE = "webcam"`
2. Show gesture, press `t` to print raw threshold values to terminal
3. Adjust constants in `gesture.py`:
   - If gesture toggles unpredictably: **increase** offset/extend thresholds by +5
   - If gesture never triggers: **decrease** thresholds by −5
4. Re-run and verify stable detection

---

## Implementation Summary

| Component | Files | Lines Added | Lines Changed |
|---|---|-::--|-::--|
| Gesture Module | `vision/gestures.py` | ~75 | 0 |
| Palmtracker Fix | `vision/palmtracker.py` | 0 | ~2 |
| Gesture Tester | `test/gesture_tester.py` | ~150 | 0 |
| **Total** | **3 files** | **~225** | **~2** |

Phase 2 is now scoped, mapped, and ready for implementation.
