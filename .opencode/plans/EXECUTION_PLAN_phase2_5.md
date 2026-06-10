# Execution Plan: Gesture V2 with Directional Palm Control

This plan details the implementation of a scale-invariant gesture engine with positional palm-based directional control, dataset collection, and evaluation.

---

## Phase Status Summary

| Phase | Deliverable | Status |
|-------|------|--------|
| Phase 1 | `vision/gestures_v2.py` | ✅ Complete |
| Phase 2 | `test/data_collector.py` | ✅ Complete |
| Phase 2 | `test/drone_fpv_analyzer.py` | ✅ Complete |
| Phase 3 | `MOTION_MODE` in `tello_handtrack.py` | ✅ Complete |
| Phase 4 | Manual data collection | ❌ Not yet done |
| Phase 4 | `test/analyze_landmarks.py` | ❌ Not yet created |
| Phase 5 | `test/evaluate_gestures.py` | ❌ Not yet created |
| Phase 5 | `benchmark_results.txt` | ❌ Not yet created |
| Phase 5 | `v2_thresholds.json` | ❌ Not yet created |
| Phase 6 | `gestures_v2` integration | ✅ Complete |

---

## Phase 1: Gesture Engine V2 Implementation

### Deliverable: `vision/gestures_v2.py`

**New Gesture Set:**
```
OPEN_PALM, CLOSED_FIST, THUMBS_UP, THUMBS_DOWN, 
PALM_UP, PALM_DOWN, PALM_LEFT, PALM_RIGHT, UNKNOWN
```

### Key Implementation Details:

1. **Normalized Coordinates (`_normalize`)**:
   - Convert normalized MediaPipe landmarks to pixel coordinates
   - Calculate hand's bounding box (min/max X, Y)
   - Normalize all distances relative to hand_box dimensions → values 0-1

2. **Gesture Classification (`classify`)**:
   - **THUMBS_UP/DOWN**: Thumb tip extends >50% beyond IP joints (normalized distance)
   - **OPEN_PALM**: All finger tip-to-tip distances >0.4 of typical rest length
   - **CLOSED_FIST**: All finger tip-to-tip distances <0.2 of typical rest length
   - **PALM_UP/DOWN**: Palm normal vector Z-axis orientation (cross-product)
   - **PALM_LEFT/RIGHT**: Hand center X position relative to image center:
     - `center_x < image_width * 0.45` → LEFT
     - `center_x > image_width * 0.55` → RIGHT
     - `0.45 <= x <= 0.55` → hover (deadzone)

3. **Palm Orientation Check (for LEFT/RIGHT)**:
   - **Visible palm**: Thumb is to the right of the hand center (from viewer's perspective)
   - Check if Thumb Tip (x > Index MCP x) for palm-facing logic

---

## Phase 2: Data Collection Tools

### Step 2: `test/data_collector.py` (Webcam Version)

**Controls:**
- Key `1`: Open Palm (center)
- Key `2`: Closed Fist (center)
- Key `3`: Thumbs Up (center)
- Key `4`: Thumbs Down (center)
- Key `5`: Palm Up (center)
- Key `6`: Palm Down (center)
- Key `7`: Palm Left (left half of frame)
- Key `8`: Palm Right (right half of frame)
- Key `q`: Quit and save
- Key `s`: Skip frame
- Key `p`: Print current thresholds

**Features:**
- 3-second capture window when gesture key pressed
- Saves frames + landmark JSON to `data/<gesture_name>`
- Auto-generates `dataset_summary.json` for analysis

### Step 3: `test/drone_fpv_analyzer.py` (FPV Stream Version)

- Identical UI to collector but ingests Tello FPV stream
- Captures gesture performance in compression/composite format
- Separate dataset directory: `data_fpv/<gesture_name>`

---

## Phase 3: Motion Mode Implementation

### Deliverable: `tello_handtrack.py`

**Configurable Constants:**
```python
MOTION_MODE = "TOGGLE"  # Options: "TOGGLE" or "SMOOTH"

# Gesture to RC speed mapping
GESTURE_RC = {
    Gesture.OPEN_PALM:     (40, 0),      # Forward
    Gesture.CLOSED_FIST:   (-40, 0),     # Backward
    Gesture.THUMBS_UP:     (0, -40),     # Up
    Gesture.THUMBS_DOWN:   (0, 40),      # Down
    Gesture.PALM_LEFT:     (-40, 0),     # Yaw Left
    Gesture.PALM_RIGHT:    (40, 0),      # Yaw Right
    Gesture.PALM_UP:       (0, 0),       # Hover
    Gesture.PALM_DOWN:     (0, 0),       # Hover
    Gesture.UNKNOWN:       (0, 0),       # Hover
}
```

### Toggle Behavior (Default):
- Drone moves at constant speed (±40) while gesture is held
- Stops immediately when gesture released or changes
- Simple on/off control

### Smooth Behavior:
- Speed scales with palm distance from deadzone center
- Left: `speed = -40 * ((0.55 - center_x) / 0.55)` → range -40 to 0
- Right: `speed = +40 * ((center_x - 0.45) / 0.55)` → range 0 to 40
- More precise control but requires finer hand positioning

### Deadzones:
- **X-axis**: `0.45 < center_x < 0.55` → hover (no accidental drift)
- **Y-axis**: `0.45 < center_y < 0.55` → hover (for up/down)
- Can configure deadzone width via `DEADZONE_WIDTH = 0.1` constant

---

## Phase 4: Data Collection & Analysis

### Step 4: Manual Data Collection (~20 minutes)

**Webcam Collection:**
```bash
cd test/
python -m data_collector
```
- Hold hand in designated zone for each key (3 seconds per gesture)
- Start with center poses, then move hand 2 feet, 4 feet away
- **Goal**: 80+ frames per gesture class (≈1000 frames total)

**FPV Collection:**
```bash
python -m drone_fpv_analyzer
```
- Connect Tello and hover
- Hold hand at comfortable distance from drone camera
- Repeat same sequence for all 8 gestures

**Directory Structure:**
```
data/
├── open_palm/
├── closed_fist/
├── thumbs_up/
├── thumbs_down/
├── palm_up/
├── palm_down/
├── palm_left/
└── palm_right/

data_fpv/
├── open_palm/
...
```

### Step 5: Threshold Analysis

**Deliverable: `test/analyze_landmarks.py`**

- Load datasets from both `data/` and `data_fpv/`
- Plot histograms of key ratios per gesture class
- Generate recommended thresholds for the new engine
- Output `v2_thresholds.json` for engine configuration

---

## Phase 5: Evaluation & Integration

### Step 6: Benchmark Script

**Deliverable: `test/evaluate_gestures.py`**

- Load collected datasets
- Run V1 (`gestures.py`) and V2 (`gestures_v2.py`) against same frames
- Calculate per-class accuracy and confusion matrix
- Compare deadzone performance in each mode
- Output: `benchmark_results.txt` with accuracy metrics

### Step 7: Integration

**Update `tello_handtrack.py`:**

1. Replace import:
   ```python
   from vision.gestures_v2 import Gesture, classify
   # Remove: from vision.gestures import Gesture, classify
   ```

2. Add MOTION_MODE to constants:
   ```python
   MOTION_MODE = "TOGGLE"  # or "SMOOTH" in code
   DEADZONE_WIDTH = 0.1
   ```

3. Update gesture display overlay:
   ```python
   GESTURE_CMD = {
       Gesture.OPEN_PALM:     ("\u2192", "FORWARD"),
       Gesture.CLOSED_FIST:   ("\u2190", "BACKWARD"),
       Gesture.THUMBS_UP:     ("\u2191", "UP"),
       Gesture.THUMBS_DOWN:   ("\u2193", "DOWN"),
       Gesture.PALM_LEFT:     ("\u2190", "LEFT"),
       Gesture.PALM_RIGHT:    ("\u2192", "RIGHT"),
       Gesture.PALM_UP:       ("\u25cb", "HOVER"),
       Gesture.PALM_DOWN:     ("\u25cb", "HOVER"),
       Gesture.UNKNOWN:       ("\u25cb", "NO TRACK"),
   }
   ```

4. Update gesture-to-RC calculation:
   ```python
   # In main loop
   speed_y, speed_z = gesture_to_rc(gesture)
   
   if MOTION_MODE == "SMOOTH" and gesture in [Gesture.PALM_LEFT, Gesture.PALM_RIGHT]:
       # Additional scaling logic based on palm position
       pass
   ```

---

## Phase 6: Implementation Priority

| Priority | Component | Reason |
|----------|-----------|--------|
| **P0** | gestures_v2.py | Core engine replacement |
| **P0** | data_collector.py | Enable dataset creation |
| **P1** | drone_fpv_analyzer.py | FPV-specific data capture |
| **P1** | MOTION_MODE in tello_handtrack.py | User-facing feature |
| **P1** | deadzone implementation | Prevent accidental drift |
| **P2** | analyze_landmarks.py | Find optimal thresholds |
| **P2** | evaluate_gestures.py | Validate improvement |

---

## Total Estimated Effort

| Phase | Time | Complexity |
|-------|------|-----------|
| 1. Engine | 3-4 hours | Medium |
| 2. Collectors | 1-2 hours | Low |
| 3. Motion Mode | 1-2 hours | Low |
| 4. Collection | 20 min (manual) | Low |
| 5. Analysis | 2-3 hours | Medium |
| 6. Benchmark | 1-2 hours | Medium |
| 7. Integration | 1 hour | Low |

**Total: ≈9 hours (including manual data collection)**

---

## Implementation Checklist

```
✓ Write vision/gestures_v2.py with normalized coordinates
  ✓ _normalize function
  ✓ gesture_to_rc function (6 basic gestures)
  ✓ PALM_LEFT/RIGHT detection via frame X position
  ✓ visible palm orientation check
  ✓ deadzone threshold handling

✓ Write tests/data_collector.py with 8 keys (1-7 and 0)
  ✓ per-gesture capture folders
  ✓ JSON frame + landmark storage
  ✓ dataset_summary.json generation

✓ Write tests/drone_fpv_analyzer.py (FPV version)
  ✓ Tello frame ingestion (avoid BGR conversion)
  ✓ identical UI to webcam collector

✓ Implement MOTION_MODE in tello_handtrack.py
  ✓ MOTION_MODE constant with TOGGLE/SMOOTH
  ✓ deadzone configuration
  ✓ configurable speed vectors per gesture

✕ Run data collection (manual)
  ✕ webcam dataset: 80+ frames per class
  ✕ fpv dataset: 80+ frames per class
  ✕ dataset directories created

✕ Write tests/analyze_landmarks.py
  ✕ histogram plotting per gesture class
  ✕ threshold recommendation output

✕ Write tests/evaluate_gestures.py
  ✕ V1 vs V2 accuracy comparison
  ✕ confusion matrix per class
  ✕ benchmark_results.txt output

✓ Update tello_handtrack.py
  ✓ Engine swap to gestures_v2
  ✓ GESTURE_RC mapping update
  ✓ GESTURE_CMD overlay update
  ✓ MOTION_MODE integration

□ Flight test
  □ Verify TOGGLE mode: crisp on/off
  □ Verify SMOOTH mode: proportional control
  □ Test deadzone prevents accidental drift
  □ Confirm PALM_DOWN stays hover
  □ Test all 6 directions
  □ Test palm visible detection
```

---

## Key Trade-offs & Design Decisions

### 1. **Position vs Rotation for L/R Detection**
- **Chosen**: Frame position (Palm center X < 0.45 → LEFT)
- **Why**: Easier to learn, more stable for FPV (camera shaking affects rotation detection more)
- **Alternative**: Wrist rotation (could be less reliable for drone stream)

### 2. **Toggle vs Smooth Motion**
- **Chosen**: Default TOGGLE, config to SMOOTH
- **Why TOGGLE first**: Less complex, immediate response, easier to debug
- **Smooth mode**: Add deadzone threshold (`DEADZONE_WIDTH = 0.1`) to prevent jitter

### 3. **Visible Palm Detection for L/R**
- **Logic**: Thumb Tip X > Index MCP X (for palm-facing camera)
- **Alternative**: Check if palm normal points toward camera
- **Risk**: Hand shadows or lighting changes could affect detection

### 4. **Threshold Strategy**
- **Approach**: Collect data → analyze → recommend → apply thresholds
- **Benefit**: Empirical approach handles different hand sizes, lighting, distances
- **Risk**: Requires careful manual collection for reliable thresholds

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| FPV compression artifacts affect gesture rec | Use drone-specific thresholds from fpv dataset |
| Palm position varies too much for reliable L/R | Deadzone (0.45-0.55) + normalize to hand_box |
| Lighting conditions break gesture detection | Collect data in varied lighting; use ratios not absolute distances |
| Hand occlusion (e.g., thumb behind palm) | Check for ≥18 of 21 landmarks before classification |

---

## Final Output Structure

```
tello-drone/
├── vision/
│   ├── gestures.py          (original, kept for comparison)
│   └── gestures_v2.py       (new engine with normalized coords)
├── test/
│   ├── data_collector.py    (webcam dataset collector)
│   ├── drone_fpv_analyzer.py (FPV dataset collector)
│   ├── analyze_landmarks.py (threshold analysis)
│   └── evaluate_gestures.py (benchmark harness)
├── data/                     (webcam dataset: 8 folders)
├── data_fpv/                 (FPV dataset: 8 folders)
├── tello_handtrack.py       (integrated with v2 engine & MOTION_MODE)
└── v2_thresholds.json       (recommended thresholds from analysis)
```

---

## Next Steps

1. **Phase 4**: Create `test/analyze_landmarks.py` (threshold analysis script)
2. **Phase 4**: Run manual data collection (~20 min, webcam + FPV, 80+ frames/gesture)
3. **Phase 5**: Create `test/evaluate_gestures.py` (benchmark V1 vs V2)
4. **Phase 5**: Generate `benchmark_results.txt` and `v2_thresholds.json`
5. **Phase 6**: Flight test and validate gestures in both motion modes
