"""Gesture classifier using hand landmarks."""

import math
from typing import List

try:
    from mediapipe.python.solutions import hand_landmark as mp_hands
    MEDIAPIPE_AVAILABLE = True
    LmIdx = mp_hands.HandLandmark
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    class LmIdx:
        WRIST = 0
        THUMB_CMC = 1
        THUMB_MCP = 2
        THUMB_IP = 3
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


class Gesture:
    CLOSE_PALM = "CLOSE_PALM"
    OPEN_PALM = "OPEN_PALM"
    POINT = "POINT"
    PINCH = "PINCH"
    THUMB_UP = "THUMB_UP"
    THUMB_DOWN = "THUMB_DOWN"
    PALM_LEFT = "PALM_LEFT"
    PALM_RIGHT = "PALM_RIGHT"
    PALM_UP = "PALM_UP"
    PALM_DOWN = "PALM_DOWN"


UNKNOWN_GESTURES = frozenset({Gesture.POINT, Gesture.PINCH, Gesture.CLOSE_PALM})


def _vector(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def _normalize(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length == 0:
        return (0, 0, 0)
    return (v[0]/length, v[1]/length, v[2]/length)


def _normalize_angle(rad):
    while rad >= math.pi:
        rad -= 2*math.pi
    while rad < -math.pi:
        rad += 2*math.pi
    return rad


def _is_finger_extended(landmarks, tip_idx, pip_idx):
    wrist = landmarks[LmIdx.WRIST]
    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    wrist_to_tip = math.sqrt((wrist.x - tip.x)**2 + (wrist.y - tip.y)**2 + (wrist.z - tip.z)**2)
    wrist_to_pip = math.sqrt((wrist.x - pip.x)**2 + (wrist.y - pip.y)**2 + (wrist.z - pip.z)**2)
    return (wrist_to_tip - wrist_to_pip) > 0.04


def _hand_bbox_angles(landmarks):
    all_points = [(lm.x, lm.y) for lm in landmarks if lm is not None and lm.x is not None and lm.y is not None]
    if len(all_points) < 2:
        return 0, 0, 0
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    if bbox_w + bbox_h == 0:
        return 0, 0, 0
    wrist_angle = math.atan2(
        landmarks[LmIdx.WRIST].y - min(ys),
        landmarks[LmIdx.WRIST].x - min(xs)
    )
    return wrist_angle, bbox_w, bbox_h


def _calc_palm_orientation(landmarks):
    mcp_positions = []
    for lm_idx in [LmIdx.INDEX_MCP, LmIdx.MIDDLE_MCP, LmIdx.RING_MCP, LmIdx.PINKY_MCP]:
        lm = landmarks[lm_idx]
        if lm.x is not None and lm.y is not None:
            mcp_positions.append((lm.x, lm.y))
    if len(mcp_positions) < 2:
        return 0.0
    mcp_xs = [p[0] for p in mcp_positions]
    mcp_ys = [p[1] for p in mcp_positions]
    dx = max(mcp_xs) - min(mcp_xs)
    dy = max(mcp_ys) - min(mcp_ys)
    if dx == 0 and dy == 0:
        return 0.0
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def _calc_thumb_palm_dist(landmarks):
    # Measure thumb proximity to the palm center (middle MCP area)
    thumb_tip = landmarks[LmIdx.THUMB_TIP]
    palm_center = landmarks[LmIdx.MIDDLE_MCP]
    ring_pip = landmarks[LmIdx.RING_PIP]
    
    dist = math.sqrt(
        (thumb_tip.x - palm_center.x)**2 +
        (thumb_tip.y - palm_center.y)**2 +
        (thumb_tip.z - palm_center.z)**2
    )
    palm_scale = math.sqrt(
        (ring_pip.x - palm_center.x)**2 +
        (ring_pip.y - palm_center.y)**2 +
        (ring_pip.z - palm_center.z)**2
    )
    if palm_scale < 1e-6:
        return dist
    return dist / palm_scale


def classify(landmarks):
    """Classify hand gesture from MediaPipe hand landmarks."""
    if not landmarks:
        return Gesture.POINT

    # Palm orientation (degrees from MCP line horizontal)
    palm_orientation_deg = _calc_palm_orientation(landmarks)

    # Extended fingers count
    num_extended = sum([
        _is_finger_extended(landmarks, LmIdx.INDEX_TIP, LmIdx.INDEX_PIP),
        _is_finger_extended(landmarks, LmIdx.MIDDLE_TIP, LmIdx.MIDDLE_PIP),
        _is_finger_extended(landmarks, LmIdx.RING_TIP, LmIdx.RING_PIP),
        _is_finger_extended(landmarks, LmIdx.PINKY_TIP, LmIdx.PINKY_PIP),
    ])

    if num_extended == 4 and palm_orientation_deg > -60:
        # Check if thumb direction overrides open palm (thumbs up / down)
        if thumb_angle_deg > 25 and bbox_h > bbox_w * 1.1:
            return Gesture.THUMB_UP
        if thumb_angle_deg < -25 and bbox_h > bbox_w * 1.1:
            return Gesture.THUMB_DOWN
        return Gesture.OPEN_PALM

    # Thumb direction using hand-centric coordinate frame
    thumb = landmarks[LmIdx.THUMB_TIP]
    wrist = landmarks[LmIdx.WRIST]
    middle_mcp = landmarks[LmIdx.MIDDLE_MCP]

    thumb_vec = (thumb.x - wrist.x, thumb.y - wrist.y, thumb.z - wrist.z)
    hand_axis = (middle_mcp.x - wrist.x, middle_mcp.y - wrist.y, middle_mcp.z - wrist.z)
    hand_axis_len = math.hypot(hand_axis[0], hand_axis[1]) + 1e-6
    norm_hand = (hand_axis[0]/hand_axis_len, hand_axis[1]/hand_axis_len, hand_axis[2]/hand_axis_len)

    # Hand "UP" vector (perpendicular to palm, pointing anatomically up)
    palm_normal = _cross(thumb_vec, hand_axis)
    palm_normal = _normalize(palm_normal)
    # Rotate hand_axis by 90 deg around palm normal to get local Y
    thumb_local_y = _cross(norm_hand, palm_normal)
    thumb_local_y = _normalize(thumb_local_y)

    # Project thumb onto hand-local Y axis
    thumb_dot_y = _dot(thumb_vec, thumb_local_y)
    thumb_dot_x = _dot(thumb_vec, norm_hand)
    thumb_angle_deg = math.degrees(math.atan2(thumb_dot_y, thumb_dot_x + 1e-6))

    # Thumb-to-palm distance
    wrist_to_mcp_dist = math.sqrt(
        (middle_mcp.x - wrist.x)**2 +
        (middle_mcp.y - wrist.y)**2 +
        (middle_mcp.z - wrist.z)**2
    )
    thumb_to_wrist_dist = math.sqrt((thumb.x - wrist.x)**2 + (thumb.y - wrist.y)**2)
    thumb_palm_dist = thumb_to_wrist_dist / max(wrist_to_mcp_dist, 1e-6)

    # Palm size
    _, bbox_w, bbox_h = _hand_bbox_angles(landmarks)

    # Palm center
    finger_tips_x = [
        landmarks[LmIdx.INDEX_TIP].x,
        landmarks[LmIdx.MIDDLE_TIP].x,
        landmarks[LmIdx.RING_TIP].x,
        landmarks[LmIdx.PINKY_TIP].x,
    ]
    valid_x = [x for x in finger_tips_x if x is not None]
    palm_center_x = sum(valid_x) / len(valid_x) if valid_x else 0.5

    # === Classification ===

    # THUMB_UP: thumb pointing up (angle_deg > 30), independent of palm facing
    if thumb_angle_deg > 25 and bbox_h > bbox_w * 1.1:
        return Gesture.THUMB_UP

    # THUMB_DOWN: thumb pointing down (angle_deg < -25)
    if thumb_angle_deg < -25 and bbox_h > bbox_w * 1.1:
        return Gesture.THUMB_DOWN

    # PALM_LEFT / PALM_RIGHT: horizontal hand
    is_horizontal = abs(palm_orientation_deg) < 45
    palm_sign = palm_normal[2] if palm_normal else 0

    if is_horizontal and num_extended >= 3:
        # Thumb position along hand's long axis
        thumb_along = _dot(thumb_vec, norm_hand)
        
        if (thumb_along < 0 and palm_sign > 0) or (thumb_along > 0 and palm_sign < 0):
            return Gesture.PALM_LEFT
        else:
            return Gesture.PALM_RIGHT

    # CLOSE_PALM / CLOSE_FIST
    if num_extended == 0:
        return Gesture.CLOSE_PALM

    # PALM_UP: Vertical hand, fingers extended, thumb near palm (thumb_palm_dist < 0.75)
    if palm_orientation_deg < -60 and num_extended >= 2:
        if thumb_palm_dist < 0.75:
            return Gesture.PALM_UP

    # PINCH: index tip near thumb tip
    index_tip = landmarks[LmIdx.INDEX_TIP]
    thumb_tip = landmarks[LmIdx.THUMB_TIP]
    index_to_thumb = math.sqrt(
        (index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2
    )
    if index_to_thumb < 0.05 and num_extended <= 1:
        return Gesture.PINCH

    return Gesture.POINT  # Default unknown
