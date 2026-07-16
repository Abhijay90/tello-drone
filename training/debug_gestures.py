"""Debug gesture classification for open_palm test."""
import sys
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone')

from unittest.mock import MagicMock
from vision.gestures_v2 import (
    classify, _normalize, _palm_orientation, _thumb_extend_ratio, _tip_to_tip_ratio,
    FINGER_TIP_SEPARATION_OPEN, THUMB_TIP_BEYOND_IP_RATIO, DEADZONE_WIDTH,
    INDEX_TIP, INDEX_MCP, MIDDLE_TIP, MIDDLE_MCP, RING_TIP, RING_MCP, PINKY_TIP, PINKY_MCP,
    Gesture
)

def make_landmarks(**coords):
    lm = []
    for i in range(21):
        m = MagicMock()
        m.x = coords.get(i, 0.5)
        m.y = coords.get(i, 0.5)
        m.z = 0.0
        lm.append(m)
    return lm

# Create an OPEN PALM gesture (right side of image for palm_right detection)
lm = make_landmarks()
for i in range(21):
    lm[i].x = 0.3  # centered
    lm[i].y = 0.3

# Wrist at center bottom
lm[0].x, lm[0].y = 0.3, 0.3

# Extended fingers pointing UP (higher y in normalized coords = closer to wrist, lower y = more extended)
for i in [5,8,9,12,13,16,17,20]:
    lm[i].x = 0.3 + ((i-9) * 0.05)
    lm[i].y = 0.1  # tips far from base

for i in [6,7,10,11,14,15,18,19]:
    lm[i].x = lm[i-1].x
    lm[i].y = 0.15  # bases between wrist and tips

# Thumb folded (close to hand)
lm[4].x = 0.35; lm[4].y = 0.25  # tip close to hand
lm[2].x = 0.34; lm[2].y = 0.28  # IP joint
lm[1].x = 0.33; lm[1].y = 0.20  # MCP

print("=== Finger separations ===")
bbox = _normalize(lm, 640, 480)[1]
box_dims = _normalize(lm, 640, 480)[2]
print(f"bounding_box: {bbox}")
print(f"box_dims: {box_dims}")
print()

checks = [
    ("index_open", INDEX_TIP, INDEX_MCP),
    ("middle_open", MIDDLE_TIP, MIDDLE_MCP),
    ("ring_open", RING_TIP, RING_MCP),
    ("pinky_open", PINKY_TIP, PINKY_MCP),
    ("thumb_extend", "thumb", None),
]

for name, tip, base in checks:
    if base is None:
        ratio = _thumb_extend_ratio(lm, 640, 480, box_dims)
        print(f"  {name}: ratio={ratio:.3f} (threshold={THUMB_TIP_BEYOND_IP_RATIO})")
    else:
        ratio = _tip_to_tip_ratio(lm, 640, 480, tip, base, box_dims)
        print(f"  {name}: ratio={ratio:.3f} (threshold={FINGER_TIP_SEPARATION_OPEN})")

palm_norm = _palm_orientation(lm, 640, 480, box_dims)
print(f"\nPalm orientation: {palm_norm:.4f} (±0.02 threshold)")

normalized = _normalize(lm, 640, 480)[0]
if normalized:
    cx = sum(p[0] for p in normalized) / len(normalized)
    print(f"Hand center X (normalized): {cx:.3f}")
    print(f"Left threshold (0.4): palm_cx={cx:.3f} {'< 0.4 LEFT' if cx < 0.4 else '>= 0.4'}")
    print(f"Right threshold (0.6): palm_cx={cx:.3f} {'> 0.6 RIGHT' if cx > 0.6 else '<= 0.6'}")

print("\n=== classify() result ===")
result = classify(lm, 640, 480)
print(f"Result: {result}")
