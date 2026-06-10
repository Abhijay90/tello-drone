# Phase 2.5: Improved Gesture Detection via Hand Geometry — Architecture Plan

**Goal:** Replace pixel-distance gesture thresholds with angle-based geometric features that are scale-invariant and robust to domain shift (webcam → FPV compression/perspective).

**Classifier Approach:** Option A — Angle-based geometric classifier (no new ML dependencies).

---

## Why Angles Over Pixels

| Problem | Pixel Thresholds (existing) | Angle Features (this plan) |
|---|---|---|
| Hand distance varies (5cm–1m) | Thresholds break — distances change ~10x | Angles stay constant |
| FPV resolution/compression blurs tips | Endpoints jitter → distances unreliable | Angles between connected joints are stable even with noisy points |
| Overhead/tilted viewpoint | Wrist→tip vectors change orientation | Finger flexion angles are relative to each other |
| Requires retuning per camera | New camera = new thresholds | Universal across cameras |

---

## Architecture

```
tello_handtrack.py ───┐
                       │
vision/gestures.py     │  (current — pixel thresholds)
vision/gestures_v2.py  │  (new — angle-based, replace this)
                       │
                       ▼
findPalm(img, drone) ──► [cx, cy, area, landmarks]
                                 │
                                 ▼
                       gestures_v2.classify(landmarks, img_w, img_h)
                                 │
                    ┌────────────┤────────────┐
                    ▼            ▼            ▼
              feature_extractor()  → normalized metrics
                    │
                    ▼
              gesture_classifier(metrics) → Gesture string
                    │
                    ▼
             tello_handtrack.py RC map → drone commands
```

### Core Functions

#### 1. `_normalize(landmarks, img_w, img_h) → np.ndarray`
- Convert pixel landmarks to a **hand-centric normalized coordinate system**
- Origin: wrist (landmark 0)
- X-axis: wrist to middle-MCP (landmark 9)
- Y-axis: perpendicular to X-axis, pointing toward pinky side
- Scale: unit of = distance(wrist, middle-MCP)
- Returns: 42-dim normalized (x, y) for 21 landmarks

#### 2. `extract_metrics(normalized_lms) → dict`
Returns a hand-invariant descriptor:

| Metric | Formula | Gesture discriminative power |
|--------|---------|------------------------------|
| `thumb_pad_open` | Angle between (wrist→index-MCP) and (wrist→thumb-MCP) | thumbs_up vs thumbs_down |
| `index_flex` | Angle at MCP9 between (PIP10→MCP9→Wrist) | extended vs folded |
| `middle_flex` | Angle at MCP13 between (PIP14→MCP13→MCP9) | extended vs folded |
| `ring_flex` | Angle at MCP17 between (PIP18→MCP17→MCP13) | extended vs folded |
| `pinky_flex` | Angle at MCP5 between (PIP6→MCP5→MCP9) | extended vs folded |
| `all_flex_sum` | Sum of index+middle+ring+pinky flexion angles | open_palm vs fist |
| `thumb_wrist_angle` | Angle between (wrist→thumb-MCP) and (wrist→index-MCP) | thumbs_up vs thumbs_down vs open |
| `palm_normal_z` | Cross product of (MCP0→MCP9) × (MCP0→MCP5), then dot with Z | palm_up vs palm_down |

#### 3. `classify(landmarks, img_w, img_h) → Gesture`
```python
norm_lms = normalize(landmarks, img_w, img_h)
metrics = extract_metrics(norm_lms)

# Decision tree (thresholds derived from data analysis, Step 3)
all_flex = metrics['all_flex_sum']

if all_flex < threshold_fist:
    gesture = Gesture.CLOSED_FIST
elif all_flex > threshold_open:
    gesture = Gesture.OPEN_PALM  # tentative — refined by thumb angle
else:
    gesture = Gesture.UNKNOWN

# Refine open_palm candidates
if gesture == Gesture.OPEN_PALM:
    if metrics['thumb_wrist_angle'] > threshold_thumb_extended:
        # Disambiguate palm_up/palm_down vs open_palm
        palm_z = metrics['palm_normal_z']
        if palm_z > threshold:
            gesture = Gesture.PALM_UP
        elif palm_z < -threshold:
            gesture = Gesture.PALM_DOWN
        else:
            gesture = Gesture.OPEN_PALM
    else:
        gesture = Gestures.THUMBS_DOWN

# Refine closed_fist candidates
if gesture == Gesture.CLOSED_FIST:
    if abs(metrics['palm_normal_z']) < threshold:
        gesture = Gesture.THUMBS_UP  # thumb sticking out of fist
```

#### 4. `get_thumb_distance(landmarks, img_w, img_h) → float`
- Returns **normalized** distance (thumb-tip-to-wrist / hand-size), not pixels

#### 5. `get_finger_count(landmarks, img_w, img_h) → (int, int)`
- Returns (extended_count, total_count) using angle thresholds instead of pixel distance

### Debug Utilities (unchanged API)
- `print_thresholds()` → prints **normalized metric values** (angles in degrees, normalized distances)
- `classify_and_debug()` → saves proof frames, prints metrics

---

## File Inventory

| File | Action | Responsibility |
|---|---|---|
| `vision/gestures_v2.py` | **NEW** | Angle-based classifier |
| `test/data_collector.py` | **NEW** | Reusable webcam data collector with gesture labeling |
| `test/drone_fpv_analyzer.py` | **NEW** | Reusable FPV gesture analyzer (auto-label + manual override) |
| `test/analyze_landmarks.py` | **NEW** | Analyze collected data, recommend thresholds |
| `test/evaluate_gestures.py` | **NEW** | Benchmark v2 vs current on both webcam and FPV datasets |
| `datasets/webcam/` | **NEW DIR** | Collected labeled samples |
| `datasets/fpv/` | **NEW DIR** | Collected FPV samples |
| `tello_handtrack.py` | **EDIT** | Import gestures_v2 |

---

## Data Collection Strategy (100-150 samples per class)

### Per-Gesture Collection Protocol
For each gesture class, the user holds it for **3-5 seconds** while the collector captures frames (~30-50 frames). Then switches gesture. Repeat for all 6 gestures. Total: ~300-900 frames.

### Gesture Key Mapping
| Key | Gesture |
|-----|------|
| `1` | open_palm |
| `2` | closed_fist |
| `3` | thumbs_up |
| `4` | thumbs_down |
| `5` | palm_up |
| `6` | palm_down |
| `q` | quit/save |
| `s` | skip frame |

### FPV Analyzer
- Same gesture key mapping
- Auto-labels using current `gestures.py` (for quick scan)
- User can re-label any frame by keypress
- Saves both image and landmarks as JSON for offline analysis
