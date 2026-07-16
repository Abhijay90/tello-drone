"""Verify the 3 previously misclassified gestures now pass."""
import sys, math
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone/src')

class Landmark:
    def __init__(self, x=0, y=0, z=0): self.x = x; self.y = y; self.z = z

def l(x, y, z): return Landmark(x, y, z)

import types
mp_hands = types.ModuleType('mp_hands')
class HandLandmark:
    WRIST = 0; THUMB_CMC = 1; THUMB_IP = 2; THUMB_TIP = 4
    INDEX_MCP = 5; INDEX_PIP = 6; INDEX_TIP = 8
    MIDDLE_MCP = 9; MIDDLE_PIP = 10; MIDDLE_TIP = 12
    RING_MCP = 13; RING_PIP = 14; RING_TIP = 16
    PINKY_MCP = 17; PINKY_PIP = 18; PINKY_TIP = 20
mp_hands.HandLandmark = HandLandmark
sys.modules['mediapipe.python.solutions'] = mp_hands
sys.modules['mediapipe'] = mp_hands
sys.modules['mediapipe.python'] = mp_hands

from gesture_classifier import classify, Gesture

def check(name, expected, actual):
    status = "PASS" if expected == actual else "FAIL"
    print(f"  [{status}] {name}: expected={expected}, got={actual}")
    return status == "PASS"

def make_landmark_list(partial_dict):
    """Make a 21-element list from a dict of {idx: landmark}."""
    lm = [None] * 21
    for k, v in partial_dict.items():
        lm[k] = v
    return lm

all_pass = True

# === TEST 1: PALM_LEFT (horizontal palm, all fingers extended, thumb on right side) ===
print("\nTEST: PALM_LEFT")
pts = {}
pts[0] = l(0.50, 0.50, 0.5)   # WRIST
pts[4] = l(0.35, 0.45, 0.5)   # THUMB_TIP (left in image for right hand = PALM_LEFT)
pts[5] = l(0.70, 0.52, 0.5)   # INDEX_MCP
pts[6] = l(0.75, 0.50, 0.5)   # INDEX_PIP
pts[9] = l(0.65, 0.48, 0.50)  # MIDDLE_MCP
pts[10] = l(0.68, 0.46, 0.5)  # MIDDLE_PIP
pts[14] = l(0.55, 0.50, 0.5)  # RING_PIP
pts[13] = l(0.55, 0.52, 0.5)  # RING_MCP
pts[17] = l(0.60, 0.52, 0.5)  # PINKY_MCP
pts[18] = l(0.60, 0.54, 0.5)  # PINKY_PIP
pts[8] = l(0.85, 0.42, 0.5)   # INDEX_TIP (extended)
pts[12] = l(0.80, 0.42, 0.5)  # MIDDLE_TIP (extended)
pts[16] = l(0.70, 0.42, 0.5)  # RING_TIP (extended)
pts[20] = l(0.62, 0.42, 0.5)  # PINKY_TIP (extended)
result = classify(make_landmark_list(pts))
all_pass &= check("PALM_LEFT", Gesture.PALM_LEFT, result)

# === TEST 2: PALM_RIGHT (horizontal palm, all fingers extended, thumb on left side) ===
print("\nTEST: PALM_RIGHT")
pts2 = {}
pts2[0] = l(0.50, 0.50, 0.5)   # WRIST
pts2[4] = l(0.65, 0.45, 0.5)   # THUMB_TIP (right in image for left hand = PALM_RIGHT)
pts2[5] = l(0.30, 0.52, 0.5)   # INDEX_MCP
pts2[6] = l(0.25, 0.50, 0.5)   # INDEX_PIP
pts2[9] = l(0.35, 0.48, 0.50)  # MIDDLE_MCP
pts2[10] = l(0.32, 0.46, 0.5)  # MIDDLE_PIP
pts2[14] = l(0.45, 0.50, 0.5)  # RING_PIP
pts2[13] = l(0.45, 0.52, 0.5)  # RING_MCP
pts2[17] = l(0.40, 0.52, 0.5)  # PINKY_MCP
pts2[18] = l(0.40, 0.54, 0.5)  # PINKY_PIP
pts2[8] = l(0.15, 0.42, 0.5)   # INDEX_TIP (extended)
pts2[12] = l(0.20, 0.42, 0.5)  # MIDDLE_TIP (extended)
pts2[16] = l(0.30, 0.42, 0.5)  # RING_TIP (extended)
pts2[20] = l(0.38, 0.42, 0.5)  # PINKY_TIP (extended)
result = classify(make_landmark_list(pts2))
all_pass &= check("PALM_RIGHT", Gesture.PALM_RIGHT, result)

# === TEST 3: THUMB_UP (vertical hand with thumb extended upward) ===
print("\nTEST: THUMB_UP")
pts3 = {}
pts3[0] = l(0.50, 0.90, 0.5)  # WRIST (bottom)
pts3[4] = l(0.50, 0.98, 0.5)  # THUMB_TIP (above wrist)
pts3[2] = l(0.48, 0.87, 0.5)  # THUMB_IP
pts3[1] = l(0.49, 0.86, 0.5)  # THUMB_CMC
pts3[5] = l(0.45, 0.70, 0.5)  # INDEX_MCP
pts3[6] = l(0.45, 0.55, 0.5)  # INDEX_PIP
pts3[8] = l(0.45, 0.20, 0.5)  # INDEX_TIP (extended upward)
pts3[9] = l(0.52, 0.70, 0.5)  # MIDDLE_MCP
pts3[10] = l(0.52, 0.55, 0.5) # MIDDLE_PIP
pts3[12] = l(0.52, 0.20, 0.5) # MIDDLE_TIP (extended upward)
pts3[13] = l(0.58, 0.70, 0.5) # RING_MCP
pts3[14] = l(0.59, 0.55, 0.5) # RING_PIP
pts3[16] = l(0.60, 0.25, 0.5) # RING_TIP (extended)
pts3[17] = l(0.64, 0.70, 0.5) # PINKY_MCP
pts3[18] = l(0.65, 0.55, 0.5) # PINKY_PIP
pts3[20] = l(0.67, 0.30, 0.5) # PINKY_TIP (extended)
result = classify(make_landmark_list(pts3))
all_pass &= check("THUMB_UP", Gesture.THUMB_UP, result)

print(f"\n{'='*60}")
print("ALL TESTS PASSED!" if all_pass else "SOME TESTS FAILED!")
print(f"{'='*60}")
