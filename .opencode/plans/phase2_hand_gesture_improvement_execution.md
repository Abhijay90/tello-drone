# Phase 2: Hand Gesture Improvement — Execution Plan

Detailed step-by-step execution plan for implementing gesture detection.

---

## Step 1: Create `vision/gestures.py`

Create the file with the following structure:

### Constants
```python
Gesture class with string constants:
  OPEN_PALM = "open_palm"
  CLOSED_FIST = "closed_fist"
  THUMBS_UP = "thumbs_up"
  THUMBS_DOWN = "thumbs_down"
  PALM_DOWN = "palm_down"
  PALM_UP = "palm_up"
  UNKNOWN = "unknown"

THUMB_EXTEND_PX = 60   # Thumb tip (4) to wrist (0)
FINGER_EXTEND_PX = 30  # Finger tip to MCP joint
PALM_ORIENT_OFFSET = 30  # Wrist Y vs MCP5 Y gap
```

### Function: `classify(landmarks, img_w, img_h) → str`
Ordered priority checks:
1. Thumb not extended + all 4 fingers extended → THUMBS_DOWN
2. Thumb extended + all 4 fingers curled → THUMBS_UP
3. Thumb extended + all 4 fingers extended:
   - wrist_y > landmarks[5].y + PALM_ORIENT_OFFSET → PALM_DOWN
   - wrist_y < landmarks[5].y - PALM_ORIENT_OFFSET → PALM_UP
   - else → OPEN_PALM
4. Thumb not extended + all 4 fingers curled → CLOSED_FIST
5. else → UNKNOWN

### Function: `get_thumb_distance(landmarks, img_w, img_h) → float`
- Euclidean distance between landmarks[4] (thumb tip) and landmarks[0] (wrist)
- Returns pixel distance at given resolution

### Function: `get_finger_count(landmarks, img_w, img_h) → (int, int)`
- Count how many of indices [8,12,16,20] (tips) are extended vs MCP [5,9,13,17]
- Returns (extended_count, total_count) as tuple (e.g., (4, 4))

### Function: `_euclidean(lm1, lm2, img_w, img_h) → float`
- Private helper
- Calculate distance between two landmarks in pixels

---

## Step 2: Edit `vision/palmtracker.py`

### Change 1: Line 48 — add landmarks variable
No change needed, `landmarks` already assigned on line 48.

### Change 2: Line 61 — add landmarks to no-hand path
```python
# Line 61 (before):
    cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
# Line 61 (after):
    cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
    return img, [cx, cy, area, landmarks]
# Remove old return on line 66
```

### Change 3: Line 62-66 — handle no-hand return
```python
# Before (lines 62-66):
else:
    cv2.putText(img, "NO HAND", (img.shape[1] // 2 - 60, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
return img, [cx, cy, area]

# After:
else:
    cv2.putText(img, "NO HAND", (img.shape[1] // 2 - 60, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img, [0, 0, 0, None]
```

---

## Step 3: Create `test/gesture_tester.py`

### File structure:

### Constants
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2, numpy as np
from vision.palmtracker import findPalm
from vision.gestures import (Gesture, classify, get_thumb_distance, 
                              get_finger_count)

DATA_SOURCE = "webcam"  # "webcam" | "drone"
WEBCAM_IDX = 0
GESTURE_DRONE_MAP = {
    Gesture.OPEN_PALM: ("→", "FORWARD"),
    Gesture.CLOSED_FIST: ("←", "BACKWARD"),
    Gesture.THUMBS_UP: ("↑", "UP"),
    Gesture.THUMBS_DOWN: ("↓", "DOWN"),
    Gesture.PALM_DOWN: ("↑", "DIST HOLD"),
    Gesture.PALM_UP: ("HOVER", "HOVER"),
    Gesture.UNKNOWN: ("○", "HOVER"),
}
# No Hand:
NO_HAND_INFO: ("—", "NO HAND")
```

### Function: `get_frame(me, cap)`
- If DATA_SOURCE == "webcam": cap.read(), return frame
- If DATA_SOURCE == "drone": me.get_frame_read().frame, return frame

### Function: `draw_top_bar(img, gesture, cx, cy, area, thumb_px, finger_count)`
- Draw 3-line text bar at top of frame
- Line 1: `Gesture: {gesture}  │ cx:{cx} cy:{cy}  │ area:{area}`
- Line 2: `THUMB: {thumb_px}px  │ FINGERS: {extended}/{total}`

### Function: `draw_gesture_table(img, active_gesture)`
- Rectangle panel in top-right corner
- List all 7 gestures + UNKNOWN (in same order as GESTURE_DRONE_MAP)
- Active gesture: green text, filled marker (●)
- Inactive: gray text, empty marker (○)
- Add drone direction arrow + text next to active gesture

### Function: `draw_status_footer(img)`
- Bottom bar: `GESTURE TESTER v2 | Press Q: Exit | T: Toggle Source`

### Function: `main()`
- Initialize cap or me based on DATA_SOURCE
- Main loop:
  1. `img = get_frame(me, cap)` — skip if None
  2. `img, info = findPalm(img)` — unpack as `info = [cx, cy, area, landmarks]`
  3. If landmarks is not None:
     - `gesture = classify(landmarks, img.shape[1], img.shape[0])`
     - `thumb_px = get_thumb_distance(landmarks, img.shape[1], img.shape[0])`
     - `finger_count = get_finger_count(landmarks, img.shape[1], img.shape[0])`
  4. Else:
     - `gesture = Gesture.UNKNOWN`
     - `thumb_px = 0.0`
     - `finger_count = (0, 4)`
  5. Draw overlays: top_bar, gesture_table, status_footer
  6. `cv2.imshow("Gesture Tester", img)`
  7. `key = cv2.waitKey(1) & 0xFF`
  8. If `key == ord('q')`: break
  9. If `key == ord('t')`: toggle DATA_SOURCE, print source name

---

## Step 4: Edit `tello_handtrack.py`

### Goal
Integrate gesture classifier into the flight loop, map gestures to RC commands, and display gesture overlay.

### Imports
Add to the top of `tello_handtrack.py` (near the other imports):

```python
from vision.gestures import Gesture, classify
```

### Change 1: Unpack landmarks from findPalm

**Current** (line ~235):
```python
img, (cx, cy, area) = findPalm(img)
```

**After:**
```python
img, (cx, cy, area, landmarks) = findPalm(img)
```

### Change 2: Replace PID control block with gesture map

**Replace** the PID control block (lines ~241-261) — the section that computes `speed_y` and `speed_z` from PID errors — with:

```python
# --- Gesture Control ---
if cx == 0 or area == 0 or landmarks is None:
    gesture = Gesture.UNKNOWN
    speed_y, speed_z = 0, 0
else:
    gesture = classify(landmarks, img.shape[1], img.shape[0])
    map = {
        Gesture.OPEN_PALM:    (40,  0),
        Gesture.CLOSED_FIST:  (-40, 0),
        Gesture.THUMBS_UP:    (0, -40),
        Gesture.THUMBS_DOWN:  (0,  40),
    }
    speed_y, speed_z = map.get(gesture, (0, 0))
```

### Change 3: Add gesture overlay

**After** the PID display text (~line ~270), add gesture overlay:

```python
# Gesture overlay
cv2.putText(img, f"Gesture: {gesture}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Direction arrow + drone command
dir_map = {
    Gesture.OPEN_PALM:     ("→", "FORWARD"),
    Gesture.CLOSED_FIST:   ("←", "BACKWARD"),
    Gesture.THUMBS_UP:     ("↑", "UP"),
    Gesture.THUMBS_DOWN:   ("↓", "DOWN"),
    Gesture.PALM_UP:       ("⊙", "HOVER"),
    Gesture.PALM_DOWN:     ("⊙", "DIST"),
    Gesture.UNKNOWN:       ("○", "NO TRACK"),
}
arrow, cmd = dir_map.get(gesture, ("○", "UNKNOWN"))
cv2.putText(img, f"{arrow} {cmd}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

### Change 4: Keep DRONE_MODE gate

The RC command line should remain mode-gated:

**Current** (line ~273):
```python
if DRONE_MODE:
    me.send_rc_control(0, speedY, speedZ, 0)
```

Leave as-is. In webcam debug mode, gesture is displayed but no RC commands are sent.

### Expected behavior per gesture

| Gesture | `speed_y` | `speed_z` | Overlay Arrow |
|---|---|-::--|
| `open_palm` | +40 | 0 | → FORWARD |
| `closed_fist` | -40 | 0 | ← BACKWARD |
| `thumbs_up` | 0 | -40 | ↑ UP |
| `thumbs_down` | 0 | +40 | ↓ DOWN |
| `palm_up` / `palm_down` | 0 | 0 | ⊙ HOVER / DIST |
| `unknown` / no hand | 0 | 0 | ○ NO TRACK |

---

## Step 5: Verification

### Test each gesture in webcam mode:

| Gesture | Expected |
|---|---|
| Open palm | `open_palm` → → FORWARD |
| Closed fist | `closed_fist` → ← BACKWARD |
| Thumbs up | `thumbs_up` → ↑ UP |
| Thumbs down | `thumbs_down` → ↓ DOWN |
| Palm facing down (wrist below MCP5) | `palm_down` → ↑ DIST HOLD |
| Palm facing up (wrist above MCP5) | `palm_up` → HOVER |
| No hand | `unknown` → — NO HAND |
| Toggle with 't' key | Switches webcam ↔ drone stream |

### Test in drone mode:
- Same gestures should display correctly over Tello FPV stream
- No drone commands sent (overlay only)

---

## Implementation Checklist

- [ ] Create `vision/gestures.py`
  - [ ] Gesture constants
  - [ ] Constants (THUMB_EXTEND_PX, FINGER_EXTEND_PX, PALM_ORIENT_OFFSET)
  - [ ] Helper function: `_euclidean`
  - [ ] Main function: `classify`
  - [ ] Export functions: `get_thumb_distance`, `get_finger_count`
- [ ] Edit `vision/palmtracker.py`
  - [ ] Line 66: append landmarks to return
  - [ ] No-hand case: append None to return
- [ ] Create `test/gesture_tester.py`
  - [ ] Constants + gesture-drone mapping dict
  - [ ] `get_frame` function
  - [ ] `draw_top_bar` function
  - [ ] `draw_gesture_table` function
  - [ ] `draw_status_footer` function
  - [ ] `main` loop with all logic
- [ ] Edit `tello_handtrack.py`
  - [ ] Add `from vision.gestures import Gesture, classify` import
  - [ ] Unpack landmarks: `img, (cx, cy, area, landmarks) = findPalm(img)`
  - [ ] Replace PID control block with gesture map
  - [ ] Add gesture overlay (gesture name + arrow/cmd)
  - [ ] Keep DRONE_MODE gate intact
- [ ] Verify all gestures in webcam mode
- [ ] Verify all gestures in drone mode
- [ ] Test 't' key toggle
- [ ] Test 'q' key exit
- [ ] Test no-hand detection

## Execution Order

1. `vision/gestures.py` — core logic, no dependencies
2. `vision/palmtracker.py` — enables landmarks for gesture module
3. `test/gesture_tester.py` — consumes modules from steps 1-2
4. `tello_handtrack.py` — consumes gestures + landmarks for flight
