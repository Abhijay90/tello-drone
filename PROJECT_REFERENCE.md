# Tello Drone Gesture Control — Project Reference

## Quick Navigation

| Category | File | Purpose |
|----------|------|---------|
| **Main** | `tello.py` | Main entry point — connects to drone, runs video feed, palm/gesture tracking, and flight controls |
| **Main** | `video.py` | Video display utility (matplotlib-based) for live drone camera stream |
| **Flight** | `drone_controller.py` | Central drone movement layer (`DroneController`) — reusable by any control system |
| **Flight** | `drone_control.py` | Unified flight control: gamepad (auto-detected, hot-plug) or keyboard fallback → `DroneController`, cv2 FPV window |
| **Flight** | `drone_flip.py` | Drone flip maneuver command script |
| **Tracking** | `tello_facetrack.py` | Face-following mode using OpenCV Haar cascades via palmtracker |
| **Tracking** | `tello_handtrack.py` | Hand-tracking flight mode — drone follows detected hand position |

## Drone Control (`drone_control.py`) — gamepad + keyboard

Unified entrypoint (replaces retired `keyboard_control.py`). Auto-detects input: gamepad if connected, else keyboard; hot-plug safe (switching sources mid-flight makes the drone hover). All movement goes through `DroneController` (`drone_controller.py`) — no input dependencies there.

**Gamepad (cine-style, verified on DualSense):**
| Input | Action |
|---|---|
| Left stick X / Y | Left-right / forward-backward (analog) |
| Right stick X / Y | Yaw / up-down |
| Cross (A, btn 0) | Takeoff |
| Circle (B, btn 1) | Land + exit |
| Options (Start, btn 9) | Emergency hover → land |

Full-stick speed = `GAMEPAD_MAX_VELOCITY` (default 50 cm/s; drone max is 100). Deadzone 0.15. Axis/button mapping constants at the top of the file for other pads. Gamepad works without window focus.

**Keyboard fallback:** W/S fwd-back, A/D left-right, R/F up-down, Q/E yaw, SPACE takeoff, L land+exit, ESC emergency. Needs the pygame control window focused (drone hovers on focus loss). `l` on the video window also lands.

Run with `python drone_control.py` (real Tello on the same Wi-Fi).

## Vision Module (`vision/`)

All vision code uses MediaPipe HandLandmarker and OpenCV.

| File | Purpose | Key Function |
|------|---------|-------------|
| `gestures_v2.py` | **Primary gesture engine** — SVM classifier (trained model) with heuristic fallback. Supports 8 gestures. | `classify(landmarks, img_w, img_h) -> Gesture` |
| `gestures.py` | Legacy gesture engine (heuristic-only, pixel-based thresholds). | `classify(landmarks, img_w, img_h)` |
| `gesture_tester_real.py` | Test script — generates simulated landmark coords for all 8 gestures and tests `gestures_v2.classify()` | Run as standalone script |
| `palmtracker.py` | Hand detection and palm center localization. Draws landmarks/connections on frame. | `findPalm(img, from_drone) -> (img, [cx, cy, area, landmarks])` |
| `facetracker.py` | Face detection using Haar cascade. Returns face bounding box + center. Used by `tello_facetrack.py`. | `findFace(img) -> (img, [[x,y], area])` |
| `video_stream.py` | Simple video stream viewer for drone FPV feed. Uses matplotlib `plt.imshow()`. | `show_vid()` |

## Gesture Reference

**8 supported gestures** in `gestures_v2.py`:

| Enum Value | Name | Detection Logic (SVM model → heuristic fallback) |
|------------|------|---------------------------------------------------|
| `Gesture.OPEN_PALM` | 1: Open Palm | All 4 fingers extended + thumb extended |
| `Gesture.CLOSED_FIST` | 2: Closed Fist | No fingers extended |
| `Gesture.THUMBS_UP` | 3: Thumbs Up | Thumb extended, 0 other fingers |
| `Gesture.THUMBS_DOWN` | 4: Thumbs Down | Thumb folded + tip below IP joint + all fingers extended |
| `Gesture.PALM_UP` | 5: Palm Up | Cross product of wrist→middle × wrist→pinky > threshold (Z > 0) |
| `Gesture.PALM_DOWN` | 6: Palm Down | Same as above but Z < 0 |
| `Gesture.PALM_LEFT` | 7: Palm Left | Hand center x < 0.4 + thumb on left edge |
| `Gesture.PALM_RIGHT` | 8: Palm Right | Hand center x > 0.6 + thumb on right edge |

**Deadzone**: Palm center x in [0.45, 0.55] range — hover zone for stable hovering.

## Training Pipeline (`training/`)

| File | Purpose | Usage |
|------|---------|-------|
| `data_collector.py` | Collect gesture frames from webcam (key 1-8) or FPV stream (`--fpv`). Saves to `data/<gesture>/sample_NNNN.json`. | `python data_collector.py [--fpv]` |
| `train_and_benchmark.py` | Train ML models RF/GB/SVM/MLP/LR/SGD on collected data. Generates confusion matrices, accuracy reports, saves best model as `.pkl`. | `python train_and_benchmark.py` |
| `benchmark_gestures.py` | Benchmark `gestures_v2.classify()` against real labeled samples. Shows per-gesture accuracy, confusion matrix, threshold overlap analysis. | `python benchmark_gestures.py` |
| `gesture_webcam_test.py` | Real-time webcam test with live accuracy stats overlay. Tracks per-class accuracy over 10 minutes. Press 'q' to quit. | `python gesture_webcam_test.py` |
| `train_lstm_model.py` | Train LSTM temporal classifier on synthetic sequences (generated from single-frame features). Outputs `.h5` model + scaler/label encoder. | `python train_lstm_model.py [--epochs 200]` |
| `analyze_data.py` | Analyze collected dataset stats (sample counts, feature ranges per gesture). | Info-only script |
| `generate_plots.py` | Plots for data analysis and benchmark results. | Uses sklearn/trained model outputs |
| `test_classifier_standalone.py` | Standalone classifier tests with simulated landmarks. | Quick unit testing |
| `palm_detect_test.py` | Tests palm detection performance independently. | Debugging / tuning |
| `drone_fpv_analyzer.py` | Analyzes FPV stream data for gesture collection quality. | Preprocessing |

## Configuration Files

| File | Purpose |
|------|---------|
| `gesture_rc_mapping.py` | Maps detected gestures to drone RC control responses (x/y/z/pitch commands) |

## Key Architecture Decisions

1. **Gesture classification uses SVM by default** — model at `training/results/best_model_svm.pkl`, loaded at startup in `gestures_v2.py`
2. **Fallback to heuristic rules** when no model file exists — matches original `gestures_v2` threshold logic
3. **Landmark coordinate system**: x,y in [0,1] (normalized), z in [-0.3, 0.3] (depth) from MediaPipe Hands model
4. **Data format**: Each sample is a JSON file containing base64 frame data + 21-point hand landmarks per gesture class
5. **SVM features** (12 total): thumb_extend_ratio, palm_orientation, palm_center_x/y, thumb_tips_spread, wrist_angle, thumb_palm_distance, palm_size, extended_fingers, wrist_palm_distance, palm_aspect_ratio, frame_angle

## Common Commands

```bash
# Run main drone control program
python tello.py

# Collect training data (webcam)
python training/data_collector.py

# Collect training data (FPV)  
python training/data_collector.py --fpv

# Train/evaluate gesture classifiers
python training/train_and_benchmark.py

# Benchmark heuristic gesture engine accuracy
python training/benchmark_gestures.py

# Real-time webcam gesture test with live accuracy
python training/gesture_webcam_test.py

# Test simulated landmark coordinates against classifier
python vision/gesture_tester_real.py
```

## Data Directory Structure

```
data/
  open_palm/
    sample_0001.json   ← {"frame": "0001.png", "width": 640, "height": 480, "landmarks": [...], "gesture": "open_palm"}
    ...
  closed_fist/
    ...
  thumbs_up/
    ...
  ... (6 more gesture dirs)
  dataset_summary.json ← {"total_samples": N, "gesture_counts": {...}}
```

## Thresholds (`gestures_v2.py`)

| Constant | Default | Meaning |
|----------|---------|---------|
| `THUMB_TIP_SEPARATION_BEYOND_IP` | 0.35 | Thumb tip distance from IP joint (relative to palm width) |
| `FINGER_TIP_SEPARATION_OPEN_RATIO` | 0.45 | Open/pinch threshold for finger extension |
| `PALM_NORMAL_Z_THRESHOLD` | 0.02 | Cross product threshold for palm UP/DOWN orientation |
| `DEADZONE_CENTER` | 0.5 | Normalized x-position for hover center |
| `DEADZONE_WIDTH` | 0.05 | Normalized half-width of deadzone |
