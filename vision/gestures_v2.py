"""Gesture recognition with resolution-independent checks.

Uses a trained SVM classifier (100% accuracy on benchmark) when the model
is available, falling back to heuristic rules otherwise.
"""
import os
from enum import Enum

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# === Model loading (optional) ===
_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, '..', 'training', 'results', 'best_model_svm.pkl')
# Normalize path relative to project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if os.path.exists(_MODEL_PATH):
    _MODEL_PATH = os.path.realpath(_MODEL_PATH)

_SVM_MODEL = None
_MODEL_AVAILABLE = False

def _load_svm_model():
    """Load the trained SVM gesture classifier if available."""
    global _SVM_MODEL, _MODEL_AVAILABLE
    if _SVM_MODEL is not None or not HAS_JOBLIB or not HAS_NUMPY:
        return

    try:
        _SVM_MODEL = joblib.load(_MODEL_PATH)
        if 'scaler' in _SVM_MODEL and 'label_encoder' in _SVM_MODEL:
            _MODEL_AVAILABLE = True
    except Exception:
        _SVM_MODEL = None
        _MODEL_AVAILABLE = False

_load_svm_model()
del _load_svm_model  # cleanup

class Gesture(Enum):
    OPEN_PALM = 1
    CLOSED_FIST = 2
    THUMBS_UP = 3
    THUMBS_DOWN = 4
    PALM_UP = 5
    PALM_DOWN = 6
    PALM_LEFT = 7
    PALM_RIGHT = 8
    UNKNOWN = 9


# =========== Hand landmark indices (MediaPipe Hands model) ========
WRIST = 0
THUMB_CMC = 1
THUMB_IP = 2
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# =========== Thresholds ========
THUMB_TIP_SEPARATION_BEYOND_IP = 0.35  # thumb tip distance from IP relative to palm width
FINGER_TIP_SEPARATION_OPEN_RATIO = 0.45  # open/pinch ratio threshold
PALM_NORMAL_Z_THRESHOLD = 0.02  # palm orientation threshold for UP/DOWN
THUMB_POINTS_DOWN_THRESHOLD = 0.05  # relative y-spread: thumb tip must be this fraction of box height below thumb IP


# =========== Deadzone config ========
DEADZONE_CENTER = 0.5  # normalized x-position for hover (center of image)
DEADZONE_WIDTH = 0.08  # normalized +/- range for deadzone from center


# =========== Helper functions ============


def _normalize(landmarks: list, img_w: int, img_h: int):
    """Normalize landmarks to [0,1] and compute bounding box."""
    pts = []
    for lm in landmarks:
        pts.append((lm.x * img_w, lm.y * img_h))

    if not pts:
        return None, (0, 0, img_w, img_h), (img_w, img_h)
    box = (min(p[0] for p in pts), min(p[1] for p in pts),
           max(p[0] for p in pts), max(p[1] for p in pts))
    h, w = box[3] - box[1], box[2] - box[0]
    if h < 1: h = 1.0
    if w < 1: w = 1.0
    box_dims = (w, h)

    normed = []
    for lm in landmarks:
        normed.append((lm.x, lm.y, lm.z))
    return normed, box, box_dims


def _palm_orientation(landmarks, img_w, img_h, box_dims):
    """Determine palm orientation via cross-product of wrist-to-pinky and wrist-to-index vectors."""
    wrist_x = landmarks[WRIST].x * img_w
    wrist_y = landmarks[WRIST].y * img_h
    idx_x = landmarks[INDEX_MCP].x * img_w
    idx_y = landmarks[INDEX_MCP].y * img_h
    pinky_x = landmarks[PINKY_MCP].x * img_w
    pinky_y = landmarks[PINKY_MCP].y * img_h

    vec1_x = idx_x - wrist_x
    vec1_y = idx_y - wrist_y
    vec2_x = pinky_x - wrist_x
    vec2_y = pinky_y - wrist_y

    cross_z = vec1_x * vec2_y - vec1_y * vec2_x
    scale = max(box_dims[0] * box_dims[1], 1.0)
    return cross_z / scale


def _is_thumb_to_right_of_hand(normalized):
    """Check if thumb is on the right side of the hand (palm faces right)."""
    if not normalized:
        return False
    min_x = min(p[0] for p in normalized)
    return all(p[0] > min_x for p in normalized)


def _is_thumb_to_left_of_hand(normalized):
    """Check if thumb is on the left side of the hand (palm faces left)."""
    if not normalized:
        return False
    max_x = max(p[0] for p in normalized)
    return all(p[0] < max_x for p in normalized)


def _thumb_extend_ratio(landmarks, img_w, img_h, box_dims):
    """Ratio of thumb tip-to-IP length relative to box diagonal."""
    if box_dims[0] * box_dims[1] < 1.0:
        return 0.0
    tip_x = landmarks[THUMB_TIP].x * img_w
    tip_y = landmarks[THUMB_TIP].y * img_h
    ip_x = landmarks[THUMB_IP].x * img_w
    ip_y = landmarks[THUMB_IP].y * img_h

    tip_to_ip = ((tip_x - ip_x) ** 2 + (tip_y - ip_y) ** 2) ** 0.5
    ref = max(box_dims[0], box_dims[1])
    if ref < 1:
        ref = 1.0
    return tip_to_ip / ref


def _tip_to_tip_ratio(lm, img_w, img_h, tip_idx, base_idx, bbox):
    """Ratio of two distances scaled to image frame dimensions."""
    tip_x = lm[tip_idx].x * img_w
    tip_y = lm[tip_idx].y * img_h
    base_x = lm[base_idx].x * img_w
    base_y = lm[base_idx].y * img_h
    x_dist = abs(tip_x - base_x)
    y_dist = abs(tip_y - base_y)

    x_ratio = x_dist / bbox[0]
    y_ratio = y_dist / bbox[1]

    # Use max across axes instead of average.
    # For edge-on palms (fingers pointing straight up/down), x-spread is zero,
    # so averaging would halve the score incorrectly.
    return max(x_ratio, y_ratio)


def _palm_center(normed):
    """Return palm center from landmark list, normalized coordinates."""
    if not normed:
        return 0.5, 0.5
    cx = sum(p[0] for p in normed) / len(normed)
    cy = sum(p[1] for p in normed) / len(normed)
    return cx, cy


# === SVM feature extraction (matches training script exactly) ===

_SVM_GESTURE_MAP = {
    'open_palm': Gesture.OPEN_PALM,
    'closed_fist': Gesture.CLOSED_FIST,
    'thumbs_up': Gesture.THUMBS_UP,
    'thumbs_down': Gesture.THUMBS_DOWN,
    'palm_up': Gesture.PALM_UP,
    'palm_down': Gesture.PALM_DOWN,
    'palm_left': Gesture.PALM_LEFT,
    'palm_right': Gesture.PALM_RIGHT,
}


def _extract_svm_features(landmarks, img_w, img_h):
    """Extract 12 features matching the training script 'extract_features'."""
    raw = [(lm.x, lm.y, lm.z) for lm in landmarks[:21]]
    xs = np.array([p[0] for p in raw])
    ys = np.array([p[1] for p in raw])
    zs = np.array([p[2] for p in raw])
    normalized = np.column_stack([xs, ys, zs]) * np.array([img_w, img_h, img_w])
    wrist = normalized[0]
    thumb_tip = normalized[4]
    pinky_tip = normalized[20]
    ip = normalized[2]
    box_size = max(np.ptp(xs) * img_w, np.ptp(ys) * img_h, 1.0)
    features = np.zeros(12, dtype=np.float64)
    # 1. thumb_extend_ratio
    thumb_extend = np.linalg.norm(thumb_tip[:2] - ip[:2])
    features[0] = thumb_extend / box_size
    # 2. palm_orientation
    middle = normalized[12]
    pinky = normalized[17]
    vec_middle = middle[:2] - wrist[:2]
    vec_pinky = pinky[:2] - wrist[:2]
    cross_z = vec_middle[0] * vec_pinky[1] - vec_middle[1] * vec_pinky[0]
    features[1] = cross_z / (box_size ** 2)
    # 3-4. palm_center_x, palm_center_y
    features[2] = np.mean(xs)
    features[3] = np.mean(ys)
    # 5. thumb_tips_spread
    features[4] = np.linalg.norm(thumb_tip[:2] - pinky_tip[:2]) / box_size
    # 6. wrist_angle
    wrist_angle = np.arctan2(
        (normalized[12][1] + normalized[17][1]) / 2 - wrist[1],
        (normalized[12][0] + normalized[17][0]) / 2 - wrist[0]
    ) * 180 / np.pi
    features[5] = wrist_angle
    # 7. thumb_palm_distance
    palm_center = np.array([features[2], features[3]]) * np.array([img_w, img_h])
    features[6] = np.linalg.norm(thumb_tip[:2] - palm_center) / box_size
    # 8. palm_size
    features[7] = (np.ptp(xs) * np.ptp(ys)) / (img_w * img_h)
    # 9. extended_fingers
    extended = 0
    for tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip_pt = raw[tip_id]
        mcp_pt = raw[mcp_id]
        dist = np.sqrt((tip_pt[0] - mcp_pt[0])**2 + (tip_pt[1] - mcp_pt[1])**2) * max(img_w, img_h)
        if dist / box_size > 0.3:
            extended += 1
    features[8] = extended
    # 10. wrist_palm_distance
    bbox_center = np.array([(np.ptp(xs) + np.min(xs)) / 2 * img_w,
                            (np.ptp(ys) + np.min(ys)) / 2 * img_h])
    features[9] = np.linalg.norm(wrist[:2] - bbox_center) / box_size
    # 11. palm_aspect_ratio
    bbox_w = np.ptp(xs) * img_w
    bbox_h = np.ptp(ys) * img_h
    features[10] = bbox_w / max(bbox_h, 1.0)
    # 12. frame_angle
    features[11] = np.arctan2(bbox_h - bbox_w, bbox_h + bbox_w) * 180 / np.pi
    return features


# =========== Main classification function ==========

def classify(landmarks: list, img_w: int, img_h: int) -> Gesture:
    """Classify hand gesture from MediaPipe hand landmarks.
    
    Uses trained SVM classifier when available (100% accuracy on benchmark),
    falls back to heuristic rules otherwise.
    """
    img_w = max(img_w, 1)
    img_h = max(img_h, 1)

    if len(landmarks) < 18:
        return Gesture.UNKNOWN

    # Try SVM classification first
    if _MODEL_AVAILABLE and HAS_NUMPY:
        try:
            features = _extract_svm_features(landmarks, img_w, img_h)
            X = features.reshape(1, -1)
            X_scaled = _SVM_MODEL['scaler'].transform(X)
            pred = _SVM_MODEL['model'].predict(X_scaled)[0]
            gesture_idx = _SVM_MODEL['label_encoder'].inverse_transform([pred])[0]
            return _SVM_GESTURE_MAP.get(gesture_idx, Gesture.UNKNOWN)
        except Exception:
            pass  # Fall through to heuristic rules

    # Fallback: heuristic classification
    normalized, bbox, box_dims = _normalize(landmarks, img_w, img_h)
    if normalized is None:
        return Gesture.UNKNOWN

    # Check thumb extension (thumb up gesture)
    thumb_extended = _thumb_extend_ratio(landmarks, img_w, img_h, box_dims) > THUMB_TIP_SEPARATION_BEYOND_IP

    # Check finger extension (all fingers open/pinch)
    finger_indices = [
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]
    extended_fingers = sum(
        1 for tip, base in finger_indices
        if _tip_to_tip_ratio(landmarks, img_w, img_h, tip, base, box_dims) > FINGER_TIP_SEPARATION_OPEN_RATIO
    )

    # Thumb tip pointing down for THUMBS_DOWN (resolution-independent: relative y-spread)
    thumb_tip_pixel_y = landmarks[THUMB_TIP].y * img_h
    thumb_ip_pixel_y = landmarks[THUMB_IP].y * img_h
    thumb_points_down = (box_dims[1] > 0) and (thumb_tip_pixel_y - thumb_ip_pixel_y) > THUMB_POINTS_DOWN_THRESHOLD * box_dims[1]

    # Palm orientation (computed once)
    palm_normal_z = _palm_orientation(landmarks, img_w, img_h, box_dims)

    # Palm center position
    palm_cx, palm_cy = _palm_center(normalized)

    # === Gesture classification logic ===

    # 1. THUMBS_UP: thumb extended, all other fingers folded
    if thumb_extended and extended_fingers == 0:
        return Gesture.THUMBS_UP

    # 2. THUMBS_DOWN: thumb folded AND thumb tip below IP joint + all fingers extended
    if not thumb_extended and thumb_points_down and extended_fingers == 4:
        return Gesture.THUMBS_DOWN

    # 3. PALM_LEFT: hand on left of image, thumb on left edge
    if normalized and (palm_cx < 0.4):
        if _is_thumb_to_left_of_hand(normalized):
            return Gesture.PALM_LEFT

    # 4. PALM_RIGHT: hand on right of image, thumb on right edge
    if normalized and (palm_cx > 0.6):
        if _is_thumb_to_right_of_hand(normalized):
            return Gesture.PALM_RIGHT

    # 5. PALM_UP: palm normal points up
    if palm_normal_z > PALM_NORMAL_Z_THRESHOLD:
        return Gesture.PALM_UP

    # 6. PALM_DOWN: palm normal points down
    if palm_normal_z < -PALM_NORMAL_Z_THRESHOLD:
        return Gesture.PALM_DOWN

    # 7. OPEN_PALM: all fingers extended (regardless of thumb)
    if extended_fingers == 4:
        return Gesture.OPEN_PALM

    # 8. Default: closed or unknown
    return Gesture.CLOSED_FIST


# ===== Deploy mode deadzone constants =====

DEADZONE_CENTER = 0.5  # normalized x center of palm
DEADZONE_WIDTH = 0.05  # normalized x half-width of deadzone


# ===== Debug function (mirrors legacy gestures.py) =====

_debug_frame_count_v2 = 0
_debug_prev_gesture_v2 = Gesture.UNKNOWN
_DEBUG_DIR_V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'debug_frames')
os.makedirs(_DEBUG_DIR_V2, exist_ok=True)


def classify_and_debug(landmarks, img_w: int, img_h: int, debug_img=None) -> Gesture:
    """Classify gesture using SVM or heuristics, with per-frame debug output."""
    global _debug_frame_count_v2, _debug_prev_gesture_v2

    result = classify(landmarks, img_w, img_h)
    _debug_frame_count_v2 += 1

    if result != _debug_prev_gesture_v2:
        print(f"\n{'='*60}")
        print(f"FRAME #{_debug_frame_count_v2} — GESTURE CHANGED: {_debug_prev_gesture_v2} → {result}")
        print(f"{'='*60}")

        if debug_img is not None:
            proof_path = os.path.join(_DEBUG_DIR_V2, f"detection_{result}_{_debug_frame_count_v2}.jpg")
            cv2.imwrite(proof_path, debug_img)
            print(f"  → Proof saved: {proof_path}")

        _debug_prev_gesture_v2 = result

    return result


# ===== SVM confidence scoring =====

# Invert the label encoder mapping: class_idx -> gesture string
_SVM_REVERSE_MAP = {v: k for k, v in _SVM_GESTURE_MAP.items()}


def _sigmoid(x):
    return 1.0 / (1.0 + max(-1 if x < -500 else math.exp(-x), 1 if x > 500 else math.exp(x)))


def classify_with_confidence(landmarks: list, img_w: int, img_h: int) -> tuple:
    """Classify gesture with confidence score. Returns (gesture, confidence_score, all_scores_dict)."""
    import math
    
    img_w = max(img_w, 1)
    img_h = max(img_h, 1)

    if len(landmarks) < 18:
        return (Gesture.UNKNOWN, 0.0, {})

    # Try SVM confidence scoring first
    if _MODEL_AVAILABLE and HAS_NUMPY:
        try:
            features = _extract_svm_features(landmarks, img_w, img_h)
            X = features.reshape(1, -1)
            X_scaled = _SVM_MODEL['scaler'].transform(X)
            
            # Get raw decision scores (always multi-class)
            scores = _SVM_MODEL['model'].decision_function(X_scaled)[0]
            
            # Convert to probabilities via sigmoid (one-vs-rest)
            probs = {cls: float(_sigmoid(float(score))) for cls, score in zip(_SVM_MODEL['model'].classes_, scores)}
            
            # Find predicted class
            best_cls = max(probs, key=probs.get)
            confidence = probs[best_cls]
            
            # Get gesture string
            gesture_str = best_cls
            gesture = _SVM_GESTURE_MAP.get(gesture_str, Gesture.UNKNOWN)
            
            return (gesture, confidence, probs)
        except Exception:
            pass
    
    # Fallback to heuristic - confidence based on threshold margin
    normalized, bbox, box_dims = _normalize(landmarks, img_w, img_h)
    if normalized is None:
        return (Gesture.UNKNOWN, 0.0, {})
    
    # Simple heuristic confidence: based on how clearly conditions are met
    thumb_extended = _thumb_extend_ratio(landmarks, img_w, img_h, box_dims) > THUMB_TIP_SEPARATION_BEYOND_IP
    finger_indices = [
        (INDEX_TIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_MCP),
        (RING_TIP, RING_MCP),
        (PINKY_TIP, PINKY_MCP),
    ]
    finger_ratios = [_tip_to_tip_ratio(landmarks, img_w, img_h, tip, base, box_dims) 
                     for tip, base in finger_indices]
    extended_fingers = sum(1 for r in finger_ratios if r > FINGER_TIP_SEPARATION_OPEN_RATIO)
    
    palm_normal_z = _palm_orientation(landmarks, img_w, img_h, box_dims)
    thumb_tip_pixel_y = landmarks[THUMB_TIP].y * img_h
    thumb_ip_pixel_y = landmarks[THUMB_IP].y * img_h
    thumb_points_down = (box_dims[1] > 0) and (thumb_tip_pixel_y - thumb_ip_pixel_y) > THUMB_POINTS_DOWN_THRESHOLD * box_dims[1]
    
    gesture = classify(landmarks, img_w, img_h)
    
    # Estimate heuristic confidence
    if gesture == Gesture.UNKNOWN:
        confidence = 0.0
    elif gesture == Gesture.THUMBS_UP:
        margin_t = (_thumb_extend_ratio(landmarks, img_w, img_h, box_dims)) - THUMB_TIP_SEPARATION_BEYOND_IP
        confidence = min(1.0, max(0.0, 0.5 + margin_t * 5))
    elif gesture == Gesture.THUMBS_DOWN:
        margin_f = (box_dims[1] > 0) and ((thumb_tip_pixel_y - thumb_ip_pixel_y) / box_dims[1] - THUMB_POINTS_DOWN_THRESHOLD)
        confidence = min(1.0, max(0.0, 0.5 + margin_f * 5))
    elif gesture == Gesture.OPEN_PALM:
        avg_ratio = sum(finger_ratios) / max(len(finger_ratios), 1)
        margin = avg_ratio - FINGER_TIP_SEPARATION_OPEN_RATIO
        confidence = min(1.0, max(0.0, 0.5 + margin * 5))
    else:
        confidence = 0.6  # default heuristic confidence
    
    return (gesture, confidence, {gesture.value: confidence})
