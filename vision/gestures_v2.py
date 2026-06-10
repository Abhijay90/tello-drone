"""Gesture recognition with resolution-independent checks."""
from enum import Enum


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
CLOSED_FIST_SPREAD = 80  # max pixel distance for 'closed' (resolution dependent)


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


# =========== Main classification function ==========

def classify(landmarks: list, img_w: int, img_h: int) -> Gesture:
    """Classify hand gesture from MediaPipe hand landmarks."""
    img_w = max(img_w, 1)
    img_h = max(img_h, 1)

    if len(landmarks) < 18:
        return Gesture.UNKNOWN

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

    # Thumb tip pointing down for THUMBS_DOWN (pixel coords: larger y = lower)
    thumb_tip_pixel_y = landmarks[THUMB_TIP].y * img_h
    thumb_ip_pixel_y = landmarks[THUMB_IP].y * img_h
    thumb_points_down = thumb_tip_pixel_y > thumb_ip_pixel_y + CLOSED_FIST_SPREAD

    # Palm orientation
    palm_normal_z = _palm_orientation(landmarks, img_w, img_h, box_dims)

    # Palm center position
    thumb_ip_pixel_y = landmarks[THUMB_IP].y * img_h
    thumb_tip_pixel_y = landmarks[THUMB_TIP].y * img_h
    thumb_points_down = thumb_tip_pixel_y > thumb_ip_pixel_y + CLOSED_FIST_SPREAD
    # Use relative y-spread: thumb_tip must be at least 5% of box height below thumb_ip
    thumb_points_down_rel = (box_dims[1] > 0) and (thumb_tip_pixel_y - thumb_ip_pixel_y) > 0.05 * box_dims[1]

    # Palm orientation
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
