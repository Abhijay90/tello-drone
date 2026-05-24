# Phase 1: Detect & Verify Hands

**Goal:** Implement and verify palm detection using HSV skin-color segmentation + OpenCV Haar Cascade in isolation.

## 📁 Files to Be Created

| File | Purpose |
|---|---|
| `vision/haarcascade_hand.xml` | Haar cascade model for hand detection |
| `vision/palmtracker.py` | Vision module: `findPalm()` function |
| `test/palm_detect_test.py` | Standalone webcam test for validation |

## 🔧 Implementation Steps

### 1. `vision/haarcascade_hand.xml`
- Download from raw content: `https://raw.githubusercontent.com/Balaje/OpenCV/master/haarcascades/hand.xml`
- Save to `vision/` directory.

### 2. `vision/palmtracker.py`
- **Function `findPalm(img)`:**
  1. **Cascade:** Load `haarcascade_hand.xml`, detect on grayscale image.
  2. **HSV Skin-Color:** Two range masks → combine with `bitwise_or`.
  3. **Contours:** Find largest on HSV mask.
  4. **Merge:** Combine bounding boxes if both detect.
  5. **Draw:** Green box + center crosshair.
  6. **Return:** `(img_overlay, [center_x, center_y, area])` or `(img, [0,0,0])`.

### 3. `test/palm_detect_test.py`
- **Source:** `cv2.VideoCapture(0)`.
- **Resolution:** Resize to `640x480`.
- **Loop:** Get frame → call `findPalm()` → display bounding box, FPS, status (`DETECTED` / `NO HAND`).
- **Terminal:** Print `(x, y, area)`.
- **Exit:** Press `q`.

## ✅ Success Criteria
- Hand detected consistently in normal lighting.
- Bounding box tracks movement smoothly.
- No false positives from similar skin-toned backgrounds.
- Runs >30 FPS on standard CPU.
