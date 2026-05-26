# Phase 1: Detect & Verify Hands

**Status:** ✅ Implemented

## 📁 Implementation Overview

Phase 1 established real-time palm detection and landmark extraction using MediaPipe. It is now fully integrated into the project and verified against standard webcam conditions.

### Files Created / Modified
| File | Status | Purpose |
|---|-|---|
| `requirement.txt` | ✅ Updated | Added `mediapipe` dependency (now in use) |
| `vision/palmtracker.py` | ✅ Created | Vision module using `mediapipe.solutions.hands` |
| `test/palm_detect_test.py` | ✅ Created | Standalone webcam verifier for bounding box, center, and area output |

### Implementation Details

Instead of the originally planned Haar Cascade + HSV Skin-Color segmentation, MediaPipe Hands was adopted for its robustness:
- **Reliability:** Performs consistently across varying skin tones and lighting without HSV range tuning.
- **Landmarks:** Extracts 21 hand points per frame; wrist landmark (index 0) serves as the stable palm center `(cx, cy)`.
- **Bounding Box & Area:** Computed from the min/max X/Y coordinates of all 21 landmarks. Area is used in Phase 2 as a distance proxy (larger area = closer hand).

### Verification Results
- Hand detection remains stable without flickering in standard indoor lighting.
- Bounding box accurately tracks hand movement.
- Zero false positives from skin-toned backgrounds during testing.
- Performs comfortably at >30 FPS, with headroom remaining for Phase 2 flight loop processing.

---

## Plan Mode Transition (v0.0.0 → v0.0.1)

**Status:** In Progress

### What Changed
- Transiently switched to Plan Mode (READ-ONLY) to construct a migration plan from MediaPipe Solution API → Task API

### What Stayed the Same
- `vision/palmtracker.py` — unchanged
- `test/palm_detect_test.py` — unchanged
- `requirement.txt` — unchanged
- All other project files — unchanged

### Reason
- Construct a well-formed migration plan before making changes


---

## Migration: Solution API → Task API (palmtracker.py)

**Status:** ✅ **Complete**

### What Worked (Final Implementation)

The migration was completed successfully by switching to `RunningMode.IMAGE` (IMAGE mode) instead of VIDEO mode, which avoids the callback requirement conflict. The key decisions and outcomes:

| Aspect | Details |
|---|---|
| **Model Path** | `tello-drone/models/hand_landmarker.task` — already present. Resolved via `os.path.dirname(os.path.dirname(SCRIPT_DIR))` to avoid relative-path issues during import from different directories. |
| **Running Mode** | `VisionTaskRunningMode.IMAGE` — allows synchronous detection via `detect()` without requiring a result_callback, unlike VIDEO mode which mandates callbacks or raises `ValueError`. |
| **Detection Call** | `mp_hands.detect(mp_image)` — synchronous method that returns a `HandLandmarkerResult` object directly. No async/callback machinery needed. Maintains the same `(img, [cx, cy, area])` return contract as the original. |
| **HAND_CONNECTIONS** | `HandLandmarksConnections.HAND_CONNECTIONS` from `mediapipe.tasks.python.vision.hand_landmarker`. All 21 landmark indices preserved (0=WRIST, 1-4=THUMB, etc.). |
| **Drawing** | Manual `cv2` drawing functions (`_draw_landmarks`, `_draw_connections`). Removed reliance on `mp_drawing` built-in utils. |
| **State Management** | No changes. `findPalm()` still returns `(img, [cx, cy, area])` directly — no module-level `_last_result` needed. |
| **Test File** | **No changes needed.** `test/palm_detect_test.py` interface contract untouched. |

### What Didn't Work (and was adjusted)

| Original Plan | Issue | Fix |
|---|---|---|
| `detect_for_numpy()` method | Removed in MediaPipe 0.10.35 Task API | Use `detect(mp.Image)` with `ImageFormat.SRGB` |
| VIDEO mode + callback | `ValueError: The vision task is in video mode, a user-defined result callback should be provided` or `should not be provided` depending on mode | Switched to IMAGE mode + synchronous `detect()`. No state needed. |
| `BaseOptions` from `mediapipe.tasks.python.core` | ImportError: not found in this package layout | Use `from mediapipe.tasks import BaseOptions` (root-level tasks namespace) |
| Relative model path `MODEL_PATH` | Resolved to wrong directory when imported as module (e.g., `vision/../models/` → `/home/abhikun/Desktop/models/`) | Used `os.path.dirname(os.path.dirname(SCRIPT_DIR))` anchored to source file location |

### Verification
- ✅ Import successful — no errors
- ✅ `HandLandmarker` loaded and model loaded without issues
- ✅ `findPalm(blank_img)` returns `[0, 0, 0]` — correct no-hand result
- ✅ Returns `(img_with_drawings, [cx, cy, area])` — matches original contract

### Files to Modify
| File | Change |
|---|---|
| `vision/palmtracker.py` | ✅ Fully migrated to Task API with IMAGE mode |
| `models/hand_landmarker.task` | ✅ Already in place |
| `test/palm_detect_test.py` | ✅ No change needed |
| `requirement.txt` | ✅ No change needed |

