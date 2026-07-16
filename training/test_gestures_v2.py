"""Test gesture classification with proper coordinate design."""
import sys
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone')

from unittest.mock import MagicMock
from vision.gestures_v2 import (
    Gesture, classify, _normalize, _palm_orientation,
    _thumb_extend_ratio, _tip_to_tip_ratio,
    INDEX_TIP, INDEX_MCP, MIDDLE_TIP, MIDDLE_MCP, RING_TIP, RING_MCP, PINKY_TIP, PINKY_MCP,
    FINGER_TIP_SEPARATION_OPEN_RATIO, THUMB_TIP_SEPARATION_BEYOND_IP,
    WRIST, THUMB_IP, THUMB_TIP,
)

def make_landmarks(coords):
    """Create landmark list from {index: (x, y)} dict."""
    lm = [MagicMock() for _ in range(21)]
    for k, (x, y) in coords.items():
        lm[k].x = x
        lm[k].y = y
        lm[k].z = 0.0
    return lm

def check_finger_extend(lm, img_w, img_h, tip_idx, base_idx, label):
    """Return (is_extended, ratio) for a finger."""
    nm, bbox, box_dims = _normalize(lm, img_w, img_h)
    if box_dims[0] < 1: box_dims = (bbox[1]-bbox[0], bbox[3]-bbox[2])
    ratio = _tip_to_tip_ratio(lm, img_w, img_h, tip_idx, base_idx, box_dims)
    return ratio > FINGER_TIP_SEPARATION_OPEN_RATIO, ratio

def check_thumb_extend(lm, img_w, img_h):
    nm, bbox, box_dims = _normalize(lm, img_w, img_h)
    ratio = _thumb_extend_ratio(lm, img_w, img_h, box_dims)
    return ratio > THUMB_TIP_SEPARATION_BEYOND_IP, ratio

def check_palm_orient(lm, img_w, img_h):
    nm, bbox, box_dims = _normalize(lm, img_w, img_h)
    zz = _palm_orientation(lm, img_w, img_h, box_dims)
    return zz

def count_extended(lm, img_w, img_h):
    nm, bbox, box_dims = _normalize(lm, img_w, img_h)
    count = 0
    for tip, base in [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                      (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]:
        ratio = _tip_to_tip_ratio(lm, img_w, img_h, tip, base, box_dims)
        if ratio > FINGER_TIP_SEPARATION_OPEN_RATIO:
            count += 1
    return count


# ==================================================================
# TEST 1: OPEN_PALM
# Edge-on palm (palm_z ~0), all 4 fingers extended, thumb folded
# ==================================================================
print("=== TEST 1: OPEN_PALM ===")
coords = {}
# Wrist at bottom center
coords[WRIST] = (0.50, 0.85)
# Thumb folded (near wrist, not extended)
coords[1] = (0.48, 0.80)  # thumb_CMC
coords[2] = (0.47, 0.78)  # thumb_MP
coords[4] = (0.46, 0.76)  # thumb_TIP
# Extended fingers (large y-span from MCP to TIP)
coords[5] = (0.50, 0.60)  # index_MCP; coords[6], [7] fill in
coords[8] = (0.50, 0.20)  # index_TIP
coords[9] = (0.50, 0.56)  # middle_MCP; coords[10], [11] fill in
coords[12] = (0.50, 0.15)  # middle_TIP
coords[13] = (0.50, 0.52)  # ring_MCP; coords[14], [15] fill in
coords[16] = (0.50, 0.10)  # ring_TIP
coords[17] = (0.50, 0.48)  # pinky_MCP; coords[18], [19] fill in
coords[20] = (0.50, 0.05)  # pinky_TIP
# fill intermediate joints
for i in [3, 6, 7, 10, 11, 14, 15, 18, 19]:
    coords[i] = (0.50, 0.40)

lm = make_landmarks(coords)
nm, bbox, box_dims = _normalize(lm, 640, 480)
print(f"bbox: {bbox}, box_dims: {box_dims}")

for name, tip, base in [
    ("index", INDEX_TIP, INDEX_MCP),
    ("middle", MIDDLE_TIP, MIDDLE_MCP),
    ("ring", RING_TIP, RING_MCP),
    ("pinky", PINKY_TIP, PINKY_MCP),
]:
    ext, ratio = check_finger_extend(lm, 640, 480, tip, base, name)
    print(f"{name}: ratio={ratio:.4f} (threshold={FINGER_TIP_SEPARATION_OPEN_RATIO}) ext={ext}")

thumb_ext, thumb_ratio = check_thumb_extend(lm, 640, 480)
print(f"thumb: ratio={thumb_ratio:.4f} (threshold={THUMB_TIP_SEPARATION_BEYOND_IP}) ext={thumb_ext}")

palm_z = check_palm_orient(lm, 640, 480)
print(f"palm_z: {palm_z:.4f}")

result = classify(lm, 640, 480)
expected = Gesture.OPEN_PALM
print(f"result: {result}")
assert result == expected, f"FAIL: expected {expected}, got {result}"
print("PASS\n")


# ==================================================================
# TEST 2: CLOSED_FIST
# All fingers closed, thumb folded, palm edge-on
# ==================================================================
print("=== TEST 2: CLOSED_FIST ===")
coords = {}
coords[WRIST] = (0.50, 0.50)
for i in range(21):
    coords[i] = (0.50, 0.50)
# Slightly spread tips from bases (not enough to trigger open)
for i in [5,8,6,7,9,12,10,11,13,16,14,15,17,20,18,19]:
    coords[i] = (0.50, 0.47)

lm = make_landmarks(coords)
nm, bbox, box_dims = _normalize(lm, 640, 480)
print(f"bbox: {bbox}, box_dims: {box_dims}")

ext_count = count_extended(lm, 640, 480)
print(f"extended_fingers: {ext_count}")

thumb_ext, thumb_ratio = check_thumb_extend(lm, 640, 480)
print(f"thumb: ratio={thumb_ratio:.4f} ext={thumb_ext}")

palm_z = check_palm_orient(lm, 640, 480)
print(f"palm_z: {palm_z:.6f}")

result = classify(lm, 640, 480)
expected = Gesture.CLOSED_FIST
print(f"result: {result}")
assert result == expected, f"FAIL: expected {expected}, got {result}"
print("PASS\n")


# ==================================================================
# TEST 3: THUMBS_UP
# Thumb extended, all 4 fingers closed/folded
# ==================================================================
print("=== TEST 3: THUMBS_UP ===")
coords = {}
coords[WRIST] = (0.50, 0.50)
for i in range(21):
    coords[i] = (0.50, 0.50)
    coords[i] = (0.48, 0.48)
# Thumb extends far beyond IP
coords[1] = (0.48, 0.45)  # thumb_CMC
coords[2] = (0.48, 0.42)  # thumb_MP
coords[4] = (0.40, 0.35)  # thumb_TIP (far from IP)

lm = make_landmarks(coords)
nm, bbox, box_dims = _normalize(lm, 640, 480)
print(f"bbox: {bbox}, box_dims: {box_dims}")

ext_count = count_extended(lm, 640, 480)
print(f"extended_fingers: {ext_count}")

thumb_ext, thumb_ratio = check_thumb_extend(lm, 640, 480)
print(f"thumb: ratio={thumb_ratio:.4f} (>= {THUMB_TIP_SEPARATION_BEYOND_IP} means extended)")

palm_z = check_palm_orient(lm, 640, 480)
print(f"palm_z: {palm_z:.6f}")

result = classify(lm, 640, 480)
expected = Gesture.THUMBS_UP
print(f"result: {result}")
assert result == expected, f"FAIL: expected {expected}, got {result}"
print("PASS\n")


# ==================================================================
# TEST 4: THUMBS_DOWN
# Thumb folded, all 4 fingers extended, thumb points DOWN (larger y)
# ==================================================================
print("=== TEST 4: THUMBS_DOWN ===")
coords = {}
coords[WRIST] = (0.50, 0.55)
# Thumb folded and pointing downward (tip has larger y than IP)
coords[1] = (0.48, 0.52)
coords[2] = (0.47, 0.51)
coords[4] = (0.46, 0.56)  # tip below IP = points down
# Fingers extended upward (tips have smaller y than MCP)
coords[5] = (0.50, 0.50)  # index_MCP
coords[8] = (0.50, 0.20)  # index_TIP
coords[9] = (0.50, 0.48)  # middle_MCP
coords[12] = (0.50, 0.18)  # middle_TIP
coords[13] = (0.50, 0.46)  # ring_MCP
coords[16] = (0.50, 0.16)  # ring_TIP
coords[17] = (0.50, 0.44)  # pinky_MCP
coords[20] = (0.50, 0.14)  # pinky_TIP
# fill rest
for i in list(set(range(21))-set(coords.keys())):
    coords[i] = (0.50, 0.30)

lm = make_landmarks(coords)
nm, bbox, box_dims = _normalize(lm, 640, 480)
print(f"bbox: {bbox}, box_dims: {box_dims}")

ext_count = count_extended(lm, 640, 480)
print(f"extended_fingers: {ext_count}")

thumb_ext, thumb_ratio = check_thumb_extend(lm, 640, 480)
print(f"thumb: ratio={thumb_ratio:.4f} ext={thumb_ext}")

palm_z = check_palm_orient(lm, 640, 480)
print(f"palm_z: {palm_z:.6f}")

result = classify(lm, 640, 480)
expected = Gesture.THUMBS_DOWN
print(f"result: {result}")
assert result == expected, f"FAIL: expected {expected}, got {result}"
print("PASS\n")


# ==================================================================
# TEST 5: PALM_LEFT
# Hand in left side, thumb on left edge of hand
# ==================================================================
print("=== TEST 5: PALM_LEFT ===")
coords = {}
for i in range(21):
    coords[i] = (0.20, 0.50)  # left side of image
coords[4] = (0.18, 0.48)  # thumb on LEFT of hand center

lm = make_landmarks(coords)
result = classify(lm, 640, 480)
print(f"result: {result}")
assert result == Gesture.PALM_LEFT, f"FAIL: expected PALM_LEFT, got {result}"
print("PASS\n")


# ==================================================================
# TEST 6: PALM_RIGHT
# Hand in right side, thumb on right edge of hand
# ==================================================================
print("=== TEST 6: PALM_RIGHT ===")
coords = {}
for i in range(21):
    coords[i] = (0.80, 0.50)  # right side of image
coords[4] = (0.82, 0.48)  # thumb on RIGHT of hand center

lm = make_landmarks(coords)
result = classify(lm, 640, 480)
print(f"result: {result}")
assert result == Gesture.PALM_RIGHT, f"FAIL: expected PALM_RIGHT, got {result}"
print("PASS\n")


# ==================================================================
# TEST 7: UNKNOWN (fewer than 18 landmarks)
# ==================================================================
print("=== TEST 7: NO_HAND ===")
lm = [MagicMock() for _ in range(10)]
for i in range(10):
    lm[i].x = 0.5
    lm[i].y = 0.5
    lm[i].z = 0
result = classify(lm, 640, 480)
print(f"result: {result}")
assert result == Gesture.UNKNOWN, f"FAIL: expected UNKNOWN, got {result}"
print("PASS\n")


print("=" * 50)
print("ALL TESTS PASSED!")
